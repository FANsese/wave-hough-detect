#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# make_synthetic_recording.R
#
# Writes a synthetic 64-channel MEA recording in the exact input format
# expected by tables2to4_pipeline.R, so that the full three-stage pipeline
# (spike extraction -> randomized Hough transform -> wavefront fitting) can be
# run end to end without the experimental recording.
#
# The synthetic trace is not meant to reproduce the published numbers. It is a
# plausibility check: propagating wavefronts sweeping an 8 x 8 grid, with the
# same channel numbering, time-column convention and signal polarity as the
# real file.
#
# Usage:
#   Rscript make_synthetic_recording.R
#   # -> synthetic_recording.csv   (65 columns: time + Ch01 ... Ch64)
#
# Then, in tables2to4_pipeline.R, set line 23 to
#   filename = "synthetic_recording.csv"
# and run it.
# ---------------------------------------------------------------------------

## ---- configuration --------------------------------------------------------

SAMPLE_RATE_HZ <- 10000     # acquisition rate of the synthetic trace
DURATION_MS    <- 9000      # recording length, milliseconds
GRID_SIDE      <- 8         # 8 x 8 planar MEA
N_CHANNELS     <- GRID_SIDE * GRID_SIDE

# Unit of the time column written to the CSV, expressed as CSV-units per
# millisecond. The analysis pipeline divides the time column by 200
# (tables2to4_pipeline.R line 115), so:
#
#   TIME_UNITS_PER_MS = 200  -> the pipeline prints activation times in ms
#                               (this matches the original file's convention)
#   TIME_UNITS_PER_MS = 1    -> time column is already in ms; comment out
#                               line 115 of the pipeline
TIME_UNITS_PER_MS <- 200

# Wave schedule. Successive waves are separated by BEAT_INTERVAL_MS on
# average; waves that would still be propagating when the recording ends are
# dropped, exactly as happens in a real experiment.
FIRST_WAVE_MS    <- 700
BEAT_INTERVAL_MS <- 1000
BEAT_JITTER_MS   <- 60

# Wavefront geometry. The source lies outside the grid to the south, which is
# the geometry recovered from the experimental recording in the paper.
SOURCE_X0 <- -3.7
SOURCE_Y0 <- -50
SPEED     <- 0.45           # grid units per millisecond

# Extracellular field potential: a fast negative deflection followed by a
# smaller, slower positive component. `SPIKE_DECAY_MS` controls how many
# samples fall inside the deflection, and therefore how reliably the
# 0.05th-percentile threshold in the pipeline picks it up. Values below about
# 0.5 ms make the spike too narrow relative to that threshold.
SPIKE_AMP_MV   <- -1.0
SPIKE_DECAY_MS <- 0.8
SPIKE_POS_AMP  <- 0.30
SPIKE_POS_MS   <- 2.5
SPIKE_POS_DECAY_MS <- 1.8
SPIKE_HALF_MS  <- 10        # half-width of the window around each activation

# Baseline, removed by the pipeline's high-pass filter rather than by hand.
NOISE_SD_MV     <- 0.02
DRIFT_MV        <- 0.5
DRIFT_PERIOD_MS <- 4000

OUT_FILE <- "synthetic_recording.csv"
SEED     <- 20260915

## ---- helpers --------------------------------------------------------------

# Coordinates of electrode `ch` on the grid, using the same channel -> (x, y)
# mapping as the analysis pipeline (tables2to4_pipeline.R lines 82-88):
# row-major, x = row, y = column.
channel_to_xy <- function(ch, side = GRID_SIDE) {
  if (ch %% side == 0) {
    c(x = ch %/% side, y = side)
  } else {
    c(x = ch %/% side + 1, y = ch %% side)
  }
}

# Source-to-electrode distance for every electrode.
electrode_distance <- function(ch) {
  xy <- channel_to_xy(ch)
  sqrt((xy[["x"]] - SOURCE_X0)^2 + (xy[["y"]] - SOURCE_Y0)^2)
}

# Cardiac extracellular field potential. `u` is time in ms relative to the
# activation time. Extracellular recordings are negative-going, which is why
# the pipeline detects spikes with which.min() rather than which.max().
spike_shape <- function(u) {
  SPIKE_AMP_MV * exp(-(u / SPIKE_DECAY_MS)^2) +
    SPIKE_POS_AMP * exp(-((u - SPIKE_POS_MS) / SPIKE_POS_DECAY_MS)^2)
}

# One electrode trace: baseline noise + slow drift + one spike per activation.
generate_channel <- function(t_ms, act_ms) {
  y <- rnorm(length(t_ms), mean = 0, sd = NOISE_SD_MV) +
       DRIFT_MV * sin(2 * pi * t_ms / DRIFT_PERIOD_MS)
  for (a in act_ms) {
    idx <- which(t_ms >= a - SPIKE_HALF_MS & t_ms <= a + SPIKE_HALF_MS)
    if (length(idx) > 0) {
      y[idx] <- y[idx] + spike_shape(t_ms[idx] - a)
    }
  }
  y
}

## ---- build the recording --------------------------------------------------

set.seed(SEED)

t_ms <- seq(0, DURATION_MS, by = 1000 / SAMPLE_RATE_HZ)

distances <- vapply(1:N_CHANNELS, electrode_distance, numeric(1))
max_delay_ms <- max(distances) / SPEED

# Activation times at the source, stopping before the recording ends.
wave_times <- FIRST_WAVE_MS
while (TRUE) {
  nxt <- tail(wave_times, 1) +
    rnorm(1, mean = BEAT_INTERVAL_MS, sd = BEAT_JITTER_MS)
  if (nxt + max_delay_ms > DURATION_MS) break
  wave_times <- c(wave_times, nxt)
}

recording <- data.frame(time = t_ms * TIME_UNITS_PER_MS)

for (ch in 1:N_CHANNELS) {
  act <- wave_times + distances[[ch]] / SPEED
  recording[[sprintf("Ch%02d", ch)]] <- generate_channel(t_ms, act)
}

write.csv(recording, OUT_FILE, row.names = FALSE)

cat(sprintf("wrote %s\n", OUT_FILE))
cat(sprintf("  %d rows x %d columns  (%d electrodes, %.1f s at %d Hz)\n",
            nrow(recording), ncol(recording), N_CHANNELS,
            DURATION_MS / 1000, SAMPLE_RATE_HZ))
cat(sprintf("  %d waves, source activations (ms): %s\n",
            length(wave_times), paste(round(wave_times), collapse = ", ")))
cat(sprintf("  max source-to-electrode delay: %.0f ms\n", max_delay_ms))
