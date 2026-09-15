options(error = recover)
require(hash)
require(rgl)
# require(Rcmdr)

# require(Rlab)
# @params LIMIT number of needles
# @params time.max duration of the whole simulation
# @params ts start time for spike
# @params x start position
# @params u.x x direction for spike to move to, should be one of -1(up), 1(down)
# @params noise.freq how frequent does noise occur
# @returns a dataframe with 2 columns: x, ts. Each row represents one 
#          observation. with x being coordinate and ts to be the timestamp
#          of the observation

# function to pick the last element in a vector
last.element = function(x) { return(tail(x,1)) }

# function to pick the first element in a vector
first = function(x) { return(head(x,1)) }

stomach.sim.one = function(LIMIT, time.max, ts, x, u.x, p, noise.freq, miu, sigma) {
  # function to test whether the spike is still inside the 1x8 grid 
  within.range = function(x,ts) {
    return((x>=1)  && (x<=LIMIT) && (ts<=time.max))
  }
  
  # dt ~ normal distribution
  dt = rnorm(1, mean=miu, sd=sigma)
  
  while (within.range(last.element(x)+ u.x, last.element(ts)+ dt)) {
    # move in the direction specified by u.x and time difference dt
    x = append(x, last.element(x) + u.x)
    ts = append(ts, last.element(ts) + dt)
    dt = rnorm(1, mean=miu, sd=sigma)
  }
  
  # spikes missing with probability p
  x.observ = c()
  ts.observ = c()
  for (i in 1:length(x)) {
    if (runif(1,0,1) > p) {
      x.observ = append(x.observ,x[i])
      ts.observ = append(ts.observ,ts[i])
    } 
  } 
  return(list("x.observ"=x.observ, "ts.observ"=ts.observ))
}

stomach.sim = function(LIMIT, time.max, ts, x, u.x, p, noise.freq, miu, sigma,
                       gap) {
  SEED_BASE = 1000000;
  
  set.seed(SEED_BASE * LIMIT * time.max * ts * x * u.x * p * noise.freq *
             miu * sigma)
  
  signal.start.ts = ts
  all.lines = list("x.observ"=c(), "ts.observ"=c())
  while (signal.start.ts < time.max) {
    one.line = stomach.sim.one(LIMIT, time.max, signal.start.ts, 
                               x, u.x, p, noise.freq, miu, sigma)
    all.lines$x.observ = c(all.lines$x.observ, one.line$x.observ)
    all.lines$ts.observ = c(all.lines$ts.observ, one.line$ts.observ)
    signal.start.ts = signal.start.ts + rexp(1, 1/gap)
  }
  
  # Generate noise with time ~ exponential distribution
  noise.ts = 0
  while (last.element(noise.ts) < time.max) {
    noise.ts = append(noise.ts, last.element(noise.ts) + rexp(1,noise.freq))
  }
  if (length(noise.ts) > 2) {
    noise.ts = noise.ts[2:(length(noise.ts) -1)]
  } else {
    noise.ts = c()
  }
  # uniformly random noise location
  noise.x = sample(1:LIMIT,length(noise.ts),replace=T)
  
  observations = data.frame(x=c(all.lines$x.observ, noise.x),
                            ts=c(all.lines$ts.observ, noise.ts),
                            z=c(rep(1,length(all.lines$x.observ)),
                                rep(0,length(noise.x))))
  # sort the spike and noise by time
  observations = observations[order(observations$ts),]
  row.names(observations) = 1:nrow(observations)
  return(observations)
}


stomach.sim.one2d = function(LIMIT, time.max, ts, x, y, u.x, u.y, p, noise.freq, miu, sigma, ratio) {
  # function to test whether the spike is still inside the 1x8 grid 
  within.range = function(y,ts) {
    return((y>=1) && (y<=LIMIT) && (ts<=time.max))
  }
  
  
  all.lines = list("x.observ"=c(), "y.observ"=c(), "ts.observ"=c())
  while (within.range(y, ts)) {
    one.line = stomach.sim.one(LIMIT, time.max, ts, 
                               x, u.x, p, noise.freq, miu, sigma)
    all.lines$x.observ = c(all.lines$x.observ, one.line$x.observ)
    all.lines$y.observ = c(all.lines$y.observ, rep(y, length(one.line$x.observ)))
    all.lines$ts.observ = c(all.lines$ts.observ, one.line$ts.observ)
    y = y + u.y
    dt = rnorm(1, mean=ratio*miu, sd=ratio*sigma)
    ts = ts + dt
  }
  
  return(all.lines)
}


stomach.sim2d = function(limit, time.max, ts, x, y, u.x, u.y, p, noise.freq, miu, sigma,
                         gap, ratio, seed.shift = 1) {
#   SEED_BASE = 1000000;
#  SEED_BASE = 100;
 SEED_BASE = 101;
# SEED_BASE = 102;
#     SEED_BASE = 103;
#     SEED_BASE = 99993;
#    SEED_BASE = 7103; 
#    SEED_BASE = 78103; 
#     SEED_BASE = 555667; 
#   SEED_BASE = 53667;
#   SEED_BASE = 4567;
  set.seed(SEED_BASE * limit * time.max * ts * x * y * u.x * u.y * p * noise.freq *
             miu * sigma + seed.shift)
  
  signal.start.ts = ts
  all.lines2d = list("x.observ"=c(), "y.observ"=c(), "ts.observ"=c())
  while (signal.start.ts < time.max) {
    one.line2d = stomach.sim.one2d(limit, time.max, signal.start.ts, 
                                   x, y, u.x, u.y, p, noise.freq, miu, sigma, ratio)
    all.lines2d$x.observ = c(all.lines2d$x.observ, one.line2d$x.observ)
    all.lines2d$y.observ = c(all.lines2d$y.observ, one.line2d$y.observ)
    all.lines2d$ts.observ = c(all.lines2d$ts.observ, one.line2d$ts.observ)
    signal.start.ts = signal.start.ts + rchisq(1, gap, ncp=0)
  }
  
  # Generate noise with time ~ exponential distribution
  noise.ts = 0
  while (last.element(noise.ts) < time.max) {
    noise.ts = append(noise.ts, last.element(noise.ts) + rexp(1,noise.freq))
  }
  if (length(noise.ts) > 2) {
    noise.ts = noise.ts[2:(length(noise.ts) -1)]
  } else {
    noise.ts = c()
  }
  # uniformly random noise location
  noise.x = sample(1:limit,length(noise.ts),replace=T)
  noise.y = sample(1:limit,length(noise.ts),replace=T)
  
  observations = data.frame(x=c(all.lines2d$x.observ, noise.x),
                            y=c(all.lines2d$y.observ, noise.y),
                            ts=c(all.lines2d$ts.observ, noise.ts),
                            z=c(rep(1,length(all.lines2d$x.observ)),
                                rep(0,length(noise.x))))
  # sort the spike and noise by time
  observations = observations[order(observations$ts),]
  row.names(observations) = 1:nrow(observations)
  return(observations)
}

compute.theta=function(observ, z, time.max){
  
  num.observed.signal = sum(z)
  #compute lambda
  lambda = (length(z) - num.observed.signal + 0.001) / (time.max + 0.001)
  
  wave = observ[which(z %in% 1),] 
  wave.position = wave$x
  wave.time = wave$ts
  num.expected.signal = abs(last.element(wave.position) - first(wave.position)) + 1
  num.missing.signal = num.expected.signal - num.observed.signal
  #print(c(num.expected.signal,num.observed.signal))
  #compute p
  p = 0.1
  if (num.observed.signal >0){
    p = (num.missing.signal+0.01) / (num.expected.signal+0.1)
  }
  
  #compute miu
  dt = c()
  j = c()
  if (num.observed.signal > 1) {
    for (i in 1:(num.observed.signal - 1)) {
      dt[i] = wave.time[i+1] - wave.time[i]
      j[i] = abs(wave.position[i+1] - wave.position[i])
    }
  }
  miu = (sum(dt) + 0.01) / (sum(j) + 0.01)
  
  #compute sigma
  s = c()
  if (num.observed.signal > 1){
    for (i in 1:(num.observed.signal - 1)) {
      s[i] = j[i] * ((dt[i]/j[i] - miu)^2)
    }
  }
  sigma = 0.1
  if (num.observed.signal > 0){
    sigma = sqrt((sum(s) + 0.001) / ((num.observed.signal - 1) + 0.1))
  }
  #if (sigma > 1) {}
  return(list("lambda"=lambda, "p"=p, "miu"=miu, "sigma"=sigma))
}

visualize = function(theta){
  limit=30
  time.max=limit * 5
  gap = 20
  lambda = theta$lambda; p = theta$p; miu = theta$miu; sigma = theta$sigma;
  u.x = 1
  sim.result = stomach.sim(limit,time.max,1,1,u.x,p,lambda,miu,sigma,gap)
  z.true = sim.result$z
  x = sim.result$x
  ts = sim.result$ts
  observ = cbind(x, ts)
  
  hough.result = hough(ts, x)
  all.indices = 1:length(x)
  signal.guess = hough.result$signal.guess
  predict.signal.indices = unlist(signal.guess$indices)
  predict.noise.indices = setdiff(all.indices, predict.signal.indices)
  true.signal.indices = which(z.true %in% 1)
  true.noise.indices = which(z.true %in% 0)
  stomach.plot(true.signal.indices, true.noise.indices,
               predict.signal.indices, predict.noise.indices,
               ts, x, signal.guess$label)
}

stomach.plot = function(true.signal.indices, true.noise.indices,
                        predict.signal.indices, predict.noise.indices,
                        ts, x, predict.signal.label) {
  green.indices = intersect(true.signal.indices, predict.signal.indices)
  yellow.indices = intersect(true.noise.indices, predict.signal.indices)
  red.indices = intersect(true.signal.indices, predict.noise.indices)
  black.indices = intersect(true.noise.indices, predict.noise.indices)
  
  plot(ts[green.indices], x[green.indices], col="green",
       xlim=c(min(ts)-10, max(ts)+10))
  points(ts[yellow.indices], x[yellow.indices], col="darkgoldenrod3")
  points(ts[red.indices], x[red.indices], col="red")
  points(ts[black.indices], x[black.indices], col="black")
  false.negative.label = c()
  if (length(red.indices) > 0) {
    for (i in 1:length(red.indices)) {
      data.index = red.indices[i]
      line = point.angle.to.line(ts[data.index],
                                 x[data.index],
                                 hough.result$best.angle)
      false.negative.label[i] = line$debug.string
    }
  }
  identify(ts[cbind(predict.signal.indices, red.indices)], 
           x[cbind(predict.signal.indices, red.indices)], 
           cbind(predict.signal.label, false.negative.label),
           ps=8)
}

get.ts = function(n, rho, x, y) {
  return((rho - n[[1]] * x - n[[2]] * y) / n[[3]])
}

draw.plane = function(n, rho) {
  UPPER.LIMIT = 8
  LOWER.LIMIT = 1
  t11 = get.ts(n, rho, LOWER.LIMIT, LOWER.LIMIT)
  t12 = get.ts(n, rho, LOWER.LIMIT, UPPER.LIMIT)
  t21 = get.ts(n, rho, UPPER.LIMIT, LOWER.LIMIT)
  t22 = get.ts(n, rho, UPPER.LIMIT, UPPER.LIMIT)
  persp3d(c(LOWER.LIMIT,UPPER.LIMIT), c(LOWER.LIMIT,UPPER.LIMIT), 
          cbind(c(t11,t21),c(t12,t22)), add=TRUE, alpha=0.5)
}

stomach.plot2d.interactive = function(all.result, highlight.plane = -1) {
  x = all.result$x; y = all.result$y; ts = all.result$ts; 
  n1 = all.result$n1; n2 = all.result$n2; n3 = all.result$n3; 
  rhos = all.result$rhos;
  stomach.plot2d(all.result)
  rgl.viewpoint( theta = 262, phi = 8, fov = 40)
  if (highlight.plane == -1) {
    while (1) {
      is.selected = select3d(button = c("right"))
      keys = all.result$keys
      for (i in 1:length(ts)) {
        xi = x[[i]]; yi = y[[i]]; tsi = ts[[i]]
        if (is.selected(x[[i]], y[[i]], ts[[i]])) {
          key = keys[[i]]
          n = c(n1[[i]], n2[[i]], n3[[i]]); rho = rhos[[i]]
  #         draw.plane(n, rho)
          draw.plane(n, rho+0.5)
          draw.plane(n, rho-0.5)
          # print(all.result[i,])
        }
      }
      indices.on.same.plane = which(keys == key)
      points3d(x[indices.on.same.plane], 
               y[indices.on.same.plane],
               ts[indices.on.same.plane],
               col="violet", size=9)
    }
  } else if (highlight.plane >= 1) {
    plane.indices = all.result$plane.indices
    indices.on.same.plane = which(plane.indices == highlight.plane)
    i = indices.on.same.plane[[1]]
    n = c(n1[[i]], n2[[i]], n3[[i]]); rho = rhos[[i]]
    # draw.plane(n, rho+0.5)
    # draw.plane(n, rho-0.5)
          draw.plane(n, rho)
    points3d(x[indices.on.same.plane], 
             y[indices.on.same.plane],
             ts[indices.on.same.plane],
             col="violet", size=9)
  }
}
stomach.plot2d = function(all.result) {

  green.indices = which(all.result$z == 1 & all.result$prediction == 1)
  yellow.indices = which(all.result$z == 0 & all.result$prediction == 1)
  red.indices = which(all.result$z == 1 & all.result$prediction == 0)
  black.indices = which(all.result$z == 0 & all.result$prediction == 0)

  ts = all.result$ts; x = all.result$x; y = all.result$y;
  
  plot3d(x[black.indices], y[black.indices], ts[black.indices], 
  # points3d(ts[black.indices], x[black.indices], y[black.indices], 
           col="black", size=3,
           xlim=c(min(x), max(x)),
           ylim=c(min(y), max(y)),
           zlim=c(min(ts), max(ts)),
           xlab="x", ylab="y", zlab="t")
  points3d(x[yellow.indices], y[yellow.indices], ts[yellow.indices], 
           col="magenta", size=3)
  points3d(x[red.indices], y[red.indices], ts[red.indices], 
           col="red", size=3)
  points3d(x[green.indices], y[green.indices], ts[green.indices], 
         col="green", size=3)
}

visualize2d = function(theta, seed.shift){
  limit=8 #16
  time.max=limit *10
  gap = 30
  lambda = theta$lambda; p = theta$p; miu = theta$miu; sigma = theta$sigma;
  x = 1; y = 1; u.x = 1; u.y = 1; ts = 1; ratio = 2;
  sim.result.with.z = stomach.sim2d(limit, time.max, ts, x, y, u.x, u.y, p, lambda, 
                             miu, sigma, gap, ratio, seed.shift)

  prediction = sim.result.with.z$z
  true.result = cbind(sim.result.with.z, data.frame(prediction))
  # stomach.plot2d.interactive(true.result, 0)

  sim.result = sim.result.with.z[,c("x", "y", "ts")]
  hough.result = hough.plane(sim.result)

  all.result = cbind(sim.result.with.z, hough.result)
  stomach.plot2d.interactive(all.result, 0)
  green.indices = which(all.result$z == 1 & all.result$prediction == 1)
  yellow.indices = which(all.result$z == 0 & all.result$prediction == 1)
  red.indices = which(all.result$z == 1 & all.result$prediction == 0)
  black.indices = which(all.result$z == 0 & all.result$prediction == 0)
  num.true.pos = length(green.indices)
  num.false.pos = length(yellow.indices)
  num.false.neg = length(red.indices)
  num.true.neg = length(red.indices)
  num.all = nrow(all.result)
  false.pos.rate = num.false.pos / num.all;
  false.neg.rate = num.false.neg / num.all;
  print(num.all)
  return(c(false.pos.rate, false.neg.rate))
}

point.angle.to.line = function(x,y,angle) {
  distance.shift = 2
  radian = angle / 180 * pi;
  slope = tan(radian)
  distance = abs(y - x * slope) / sqrt(slope * slope + 1)
  distance.shifted = distance * distance.shift
  distance.rounded = as.integer(distance.shifted)
  key = paste(angle, distance.rounded, sep=",")
  debug.string = sprintf("ts=%0.2f,x=%d,d=%0.2f,ds=%0.2f,k=(%s)",
                         x,y,distance,distance.shifted,key)
  return(list("key"=key, "debug.string"=debug.string))
}

cross.product = function(u, v) {
  return(c(u[[2]]*v[[3]] - u[[3]]*v[[2]],
           u[[3]]*v[[1]] - u[[1]]*v[[3]],
           u[[1]]*v[[2]] - u[[2]]*v[[1]]))
}

get.step.key = function(number.to.round, step) {
  return(as.integer(number.to.round / step) * step)
}

get.key = function(rho, phi, theta) {
  rho.step = get.step.key(rho, 0.5)
  phi.step = get.step.key(phi/pi*180, 2)
  theta.step = get.step.key(theta/pi*180, 2)
  key = paste(rho.step, phi.step, theta.step, sep=",")
  return(key)
}

is.point.on.plane = function(point, normal, rho) {
  distance.to.plane = abs(dot.product(point, normal) - rho)
  return(distance.to.plane < 1)
}

find.points.on.plane = function(xs.all, ys.all, zs.all, normal, rho, 
                                unclassify.indices) {
  indices.on.plane = c()
  for (i in unclassify.indices) {
    p = c(xs.all[[i]], ys.all[[i]], zs.all[[i]])
    if (is.point.on.plane(p, normal, rho)) {
      indices.on.plane = append(indices.on.plane, i)
    }
  }
  return(indices.on.plane)
}

num.unique.detectors = function(xs, ys, points.on.plane ) {
  detectors = hash()
  for (index in points.on.plane) {
    key = paste(xs[[index]], ys[[index]])
    if (!has.key(key, detectors)) {
      detectors[[key]] = 1
    }
  }
  return(length(keys(detectors)))
}

dot.product = function (u, v) {
  return(sum(u*v))
}

hough.plane = function(data.points) {
  num.data = nrow(data.points)
  prediction = rep(0, num.data)
  n1 = rep(0, num.data)
  n2 = rep(0, num.data)
  n3 = rep(0, num.data)
  plane.indices = rep(0, num.data)
  n1.by.plane.index = c()
  n2.by.plane.index = c()
  n3.by.plane.index = c()
  rho.signed.by.plane.index = c()
  rhos = rep(0, num.data)
  plane.index = 1
  keys = rep("", num.data)
  unclassify.indices = 1:num.data
  accumulator = hash()
  threshold = 8; iter.max = 30000
  for (i in 1:iter.max) {
    unclassify.indices = which(prediction == 0)
    if (length(unclassify.indices) < 3) {break}
    sample.indices = sample(unclassify.indices, 3)
    #print(sample.indices)
    v1 = unlist(data.points[sample.indices[[1]],])
    v2 = unlist(data.points[sample.indices[[2]],])
    v3 = unlist(data.points[sample.indices[[3]],])

    n = cross.product(v2 - v1, v3 - v1)
    norm.n = sqrt(dot.product(n,n)); if (norm.n == 0) next
    n = n / norm.n
    nn = n
    if (n[[1]] != 0) {
      n = n * sign(n[[1]])
    }
    if ((n[[1]] == 0 & n[[2]] == 0) |
          (n[[2]] == 0 & n[[3]] == 0) |
          (n[[1]] == 0 & n[[3]] == 0)) {
      next
    }
    if (dot.product(n,c(0,0,1))==0) next
    rho.signed = dot.product(n,v1)
    rho = abs(rho.signed)
    phi = acos(n[[3]]); if (phi == 0) next
    theta = asin(n[[2]] / sin(phi)); 

    key = get.key(rho, phi, theta)
#     if (!has.key(key, accumulator)) { accumulator[[key]] = 0 }
#     accumulator[[key]] = accumulator[[key]] + 1
    if (!has.key(key, accumulator)) {
      accumulator[[key]] = c()
    }
    accumulator[[key]] = append(accumulator[[key]], sample.indices)
    if (length(accumulator[[key]])> threshold * 3) {
#       indices.on.plane = find.points.on.plane(data.points$x,  
#          data.points$y, data.points$ts, n, rho.signed, unclassify.indices)
      
      fit.input = cbind(as.matrix(data.points[accumulator[[key]], 2:3]),
                        rep(1, length(data.points[accumulator[[key]], 1])))
      lm.fit.result = lm.fit(fit.input, data.points[accumulator[[key]], 1])
      coef = lm.fit.result$coefficients
      n.fit = c(1, -coef[[1]], -coef[[2]]); rho.fit = coef[[3]]
      norm.n.fit = sqrt(dot.product(n.fit,n.fit))
      n.fit = n.fit/norm.n.fit; rho.fit = rho.fit/norm.n.fit
      # print(n)
      # print(n.fit)
      # print(rho.signed)
      # print(rho.fit)
      # print("break")
      indices.on.plane = find.points.on.plane(data.points$x, 
         data.points$y, data.points$ts, n.fit, rho.fit, unclassify.indices)
      num.detectors.with.signal = num.unique.detectors(data.points$x, 
         data.points$y, indices.on.plane)
      # TODO: change the number 40 to depend on limit. e.g. limit*limit*66%
      if (num.detectors.with.signal < 40) {
        # print(i)
        # print(num.detectors.with.signal)
        accumulator[[key]] = c()
        next
      }
      keys[indices.on.plane] = key; 
      prediction[indices.on.plane] = 1
#       n1[indices.on.plane] = n[[1]]
#       n2[indices.on.plane] = n[[2]]
#       n3[indices.on.plane] = n[[3]]
#       rhos[indices.on.plane] = rho.signed
      n1[indices.on.plane] = n.fit[[1]]
      n2[indices.on.plane] = n.fit[[2]]
      n3[indices.on.plane] = n.fit[[3]]
      rhos[indices.on.plane] = rho.fit
      plane.indices[indices.on.plane] = plane.index
#       n1.by.plane.index[plane.index] = n[[1]]
#       n2.by.plane.index[plane.index] = n[[2]]
#       n3.by.plane.index[plane.index] = n[[3]]
#       rho.signed.by.plane.index[plane.index] = rho.signed
      n1.by.plane.index[plane.index] = n.fit[[1]]
      n2.by.plane.index[plane.index] = n.fit[[2]]
      n3.by.plane.index[plane.index] = n.fit[[3]]
      rho.signed.by.plane.index[plane.index] = rho.fit
      plane.index = plane.index + 1
      accumulator = hash()
    }
  }
#   print(plane.indices)
# find the planes with "true" direction  
  positive.plane.indices = plane.indices[plane.indices > 0]
  max.index = mode(positive.plane.indices)
#   print(max.index)
  max.normal.vector = c(n1.by.plane.index[[max.index]],
                        n2.by.plane.index[[max.index]],
                        n3.by.plane.index[[max.index]])
  plane.indices.guess.as.noise = c()
  plane.indices.guess.as.signal = c()
  angle.between.planes = c()
  angle.threshold = 5
  for (i in 1:length(n1.by.plane.index)) {
    plane.normal.vector = c(n1.by.plane.index[[i]],
                            n2.by.plane.index[[i]],
                            n3.by.plane.index[[i]])
    radian = acos(dot.product(max.normal.vector, plane.normal.vector) / 
                  (sqrt(dot.product(max.normal.vector, max.normal.vector) *
                        dot.product(plane.normal.vector, plane.normal.vector))))
    angle.difference = radian / pi * 180
    angle.between.planes = append(angle.between.planes,min(angle.difference, 180 - angle.difference))
    if (min(angle.difference, 180 - angle.difference) > angle.threshold) {
      plane.indices.guess.as.noise = append(plane.indices.guess.as.noise, i)
    }
    else {
      plane.indices.guess.as.signal = append(plane.indices.guess.as.signal, i)
    }
  }
#   plot(sort(angle.between.planes))
  noise.indices = which(plane.indices %in% plane.indices.guess.as.noise)
  signal.indices = which(plane.indices %in% plane.indices.guess.as.signal)
  prediction[noise.indices] = 0

# find the unclassified points whose distances to the "true" planes is small  
  for (i in plane.indices.guess.as.signal) {
    unclassify.indices = which(prediction == 0)
    rho = abs(rho.signed.by.plane.index[[i]])
    phi = acos(n3.by.plane.index[[i]]);
    theta = asin(n2.by.plane.index[[i]]/ sin(phi)); 
    key = get.key(rho, phi, theta)
    indices.on.plane = find.points.on.plane(data.points$x,  
                        data.points$y, data.points$ts, c(n1.by.plane.index[[i]],
                                                         n2.by.plane.index[[i]],
                                                         n3.by.plane.index[[i]]), 
                        rho.signed.by.plane.index[[i]], unclassify.indices)
    prediction[indices.on.plane] = 1
    keys[indices.on.plane] = key
    n1[indices.on.plane] = n1.by.plane.index[[i]]
    n2[indices.on.plane] = n2.by.plane.index[[i]]
    n3[indices.on.plane] = n3.by.plane.index[[i]]
    rhos[indices.on.plane] = rho.signed.by.plane.index[[i]]
#     plane.indices[indices.on.plane] = i
  }

# find the unclassified points that can spanned a "true" plane  
  small.plane.threshold = 30
#   i = 1
  while (length(unclassify.indices) >= 3) {
    unclassify.indices = which(prediction == 0)
    i = 1
    v = unlist(data.points[unclassify.indices[[i]],])
    rho.signed = dot.product(max.normal.vector, v)
    rho = abs(rho.signed)
    phi = acos(max.normal.vector[[3]]);
    theta = asin(max.normal.vector[[2]] / sin(phi)); 
    key = get.key(rho, phi, theta)
    indices.on.plane = find.points.on.plane(data.points$x,  
         data.points$y, data.points$ts, max.normal.vector, 
         rho.signed, unclassify.indices)
    
#     num.detectors.with.signal = num.unique.detectors(data.points$x, 
#                                                      data.points$y, indices.on.plane)
#     # TODO: change the number 40 to depend on limit. e.g. limit*limit*66%
#     if (num.detectors.with.signal < 40) {
#       print(i)
#       print(num.detectors.with.signal)
#       indices.on.plane = c()
#       i = i + 1
#       next
#     }
    
    # print(length(indices.on.plane))
    if (length(indices.on.plane)>small.plane.threshold) {
      prediction[indices.on.plane] = 1
      keys[indices.on.plane] = key
      n1[indices.on.plane] = max.normal.vector[[1]]
      n2[indices.on.plane] = max.normal.vector[[2]]
      n3[indices.on.plane] = max.normal.vector[[3]]
      rhos[indices.on.plane] = rho.signed
    } else { 
      prediction[unclassify.indices[i]] = 2 
    }
  }
  noise.indices = which(prediction == 2)
  prediction[noise.indices] = 0
  
  return(data.frame(prediction, keys, n1, n2, n3, plane.indices, rhos))
}

mode = function(vector.of.integers) {
  return(as.integer(names(sort(-table(vector.of.integers))[1])))
}

hough2d = function(xs,ys,zs) {
  num.data = length(xs)
  angle.distance.to.points = hash()
  for (i in 1:num.data) {
    x = xs[[i]]
    y = ys[[i]]
    # Discretization at 10 degree intervals
    for (angle in (1:35)*5) {
      line = point.angle.to.line(x,y,angle)
      if (!has.key(line$key, angle.distance.to.points)) {
        angle.distance.to.points[[line$key]] = c()
      }
      angle.distance.to.points[[line$key]] = 
        rbind(angle.distance.to.points[[line$key]], 
              list("indices"=i, "label"=line$debug.string))
    }
  }
  
  num.points.threshold = 5
  indices.above.threshold = c()
  best.angle = -1
  for (key in keys(angle.distance.to.points)) {
    point.indices = angle.distance.to.points[[key]]
    num.points = nrow(point.indices)
    if (num.points > num.points.threshold) {
      angle.distance.pair = strsplit(key, ",")[[1]]
      angle = as.double(angle.distance.pair[[1]])
      distance = as.double(angle.distance.pair[[2]])
      indices.above.threshold = rbind(indices.above.threshold, point.indices)
      best.angle = angle
    }
  }
  return(list("signal.guess"=data.frame(indices.above.threshold),
              "best.angle"=best.angle))
}

optimality.table = function() {
  optimal.stats = c()
  
  shifts = 0
  # shifts = (10:10)/10
  # shifts = 1:100
  for (shift in shifts){
    theta.true = list("lambda"=1, "p"=0.1, "miu"=1, "sigma"=0.1)
    optimal.stat = visualize2d(theta.true, shift)
    optimal.stats = rbind(optimal.stats, optimal.stat)
    print(shift)
  }
  optimal.stats = cbind(shifts, optimal.stats)
  print(optimal.stats)
  print(c(mean(optimal.stats[,2]), mean(optimal.stats[,3])))
  print(c(sd(optimal.stats[,2]), sd(optimal.stats[,3])))
  
  # ps = (1:50)/100
  # for (p in ps){
  # theta.true = c(0.01, p, 1, 0.1)
  # optimal.stat = optimality(theta.true)
  # optimal.stats = rbind(optimal.stats, optimal.stat)
  # }
  # optimal.stats = cbind(ps, optimal.stats)
  # print(optimal.stats)
}

optimality.table()
