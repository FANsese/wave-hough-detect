options(error=recover)
require(rgl)
require(signal)

run.smoothing = TRUE
LOWESS = 1
BUTTER = 2
NOSMOOTH = 3
plot.spike = TRUE
# smooth.method = BUTTER
# smooth.method = LOWESS
smooth.method = NOSMOOTH
run_hough = TRUE
run.hybrid.optim = TRUE
run.plane.model = TRUE

if (run.smoothing) {
  # some comment
  # filename = "C:/Users/Amy Lu/Documents/My Dropbox/Research/Simulate/CM_PIN_Control_2N30_Aunor-txt.csv"
  filename = "CM_PIN_Control_2N30_Aunor-txt.csv"
 # data = read.csv(filename)
  data = data.matrix(data)
  l = ncol(data);
  f = function(str) {as.integer(substr(str,3,4))};
  channel = sapply(colnames(data)[2:l], f)
  
  time = data[,1]
  # activation.time = c()
  # all.lines = list("channel"=c(), "min.index"=c(), "ts.observ"=c())
  all.lines = list("x.observ"=c(), "y.observ"=c(), "ts.observ"=c())
  max.index = c()
  
  # time.start = proc.time()[1]
  for (smooth.method in 1) {
    # for (i in 2:ncol(data)) {
    # for (i in 11:ncol(data)) {
    for (i in 6) {
      print(paste(smooth.method, ' ', i))
      #       i = 38
      unsmooth=data[,i]*(-1)
      smooth = c()
      if (smooth.method == LOWESS) {
        smooth = lowess(1:length(unsmooth),unsmooth, 
                        f = 20/length(unsmooth),delta=1/length(unsmooth))
        max.index.gap = 500
      } else if (smooth.method == BUTTER) {
        butter.filter = butter(2, 1/500, type="high")
        smooth = filter(butter.filter, unsmooth)
        smooth$y = smooth
        smooth$x = 1:length(smooth$y)
        upper.threshold = quantile(smooth$y, 0.9995)
        max.index.gap = 500
      } else if (smooth.method == NOSMOOTH) {
        smooth$y = unsmooth
        smooth$x = 1:length(smooth$y)
        max.index.gap = 500
      }
      change.smooth = smooth
      smooth.y.mean = mean(smooth$y) 
      smooth.y.sd = sd(smooth$y)
      this.max.index = c()
      j = 0
      while (1) {
        j = j + 1
        if (smooth.method == LOWESS) {
          if (smooth$y[which.max(change.smooth$y)] <=
                (smooth.y.mean + 2 * smooth.y.sd )) {
            break
          }
        } else if (smooth.method == BUTTER) {
          if (smooth$y[which.max(change.smooth$y)] <= upper.threshold) {
            break
          }
        } else if (smooth.method == NOSMOOTH) {
          if (smooth$y[which.max(change.smooth$y)] <=
                (smooth.y.mean + 2 * smooth.y.sd )) {
            break
          }
        }
        #     all.lines$channel = c(all.lines$channel, i-1)
        if (channel[i-1]%%8 == 0){
          all.lines$x.observ = c(all.lines$x.observ, channel[i-1]%/%8)
          all.lines$y.observ = c(all.lines$y.observ, 8)
        } else {
          all.lines$x.observ = c(all.lines$x.observ, channel[i-1]%/%8 + 1)
          all.lines$y.observ = c(all.lines$y.observ, channel[i-1]%%8)
        }
        
        max.index = which.max(change.smooth$y)
        this.max.index = c(this.max.index, max.index)
        
        ts.observ = time[max.index]
        all.lines$ts.observ = c(all.lines$ts.observ, ts.observ)
        
        xs = smooth$x/10
        xs = xs[which(change.smooth$y >-100000)]
        ys = -data[,i]
        ys = ys[which(change.smooth$y >-100000)]
        plot(xs,ys, col="green", xlab="Time(ms)", ylab="Voltage(mV)",
          type="l")
        ys2 = smooth$y
        ys2 = ys2[which(change.smooth$y >-100000)]
        if (smooth.method != NOSMOOTH) {
          lines(xs,ys2,col=4)
        }
        points(this.max.index[j]/10, smooth$y[this.max.index[j]], col="red")
        dev.copy(pdf,paste('plots/smoothing', smooth.method, '_', i, '_', j,'.pdf', sep=''))
        dev.off()

        change.smooth$y[
          (max.index-max.index.gap):(max.index+max.index.gap)] = -1000000
        
      }
      
        #       print('here')
      xs = smooth$x/10
      plot(xs,-data[,i], col="green", xlab="Time(ms)", ylab="Voltage(mV)",
        type="l")
      if (smooth.method != NOSMOOTH) {
        lines(xs,smooth$y,col=4)
      }
      #       points(min.index, smooth$y[min.index], col="red")
      if (plot.spike) {
        points(this.max.index/10, smooth$y[this.max.index], col="red")
      }
      # break
     dev.copy(png,paste('plots/smoothing', smooth.method, '_', i, '.png', sep=''))
     dev.off()
    }
  }
  # time.end= proc.time()[1]
  # print('time used')
  # print(time.end-time.start)
  all.lines = as.data.frame(all.lines)
  all.lines = all.lines[order(all.lines$ts.observ),]
  all.lines$ts.observ= all.lines$ts.observ/200
  
}
print(all.lines)
