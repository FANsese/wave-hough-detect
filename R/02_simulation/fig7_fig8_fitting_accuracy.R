options(error=recover)
require(hash)
require(rgl)
require(numDeriv)
# require(optimx)
# require(minqa)

# require(Rlab)

# === 添加目录检查代码 ===
if (!dir.exists("plots")) {
  dir.create("plots", recursive = TRUE)
}
# =======================

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
last = function(x) { return(tail(x,1)) }

# function to pick the first element in a vector
first = function(x) { return(head(x,1)) }

stomach.sim.one = function(LIMIT, time.max, ts, x, u.x, p, noise.freq, miu, sigma) {
  # function to test whether the spike is still inside the 1x8 grid 
  within.range = function(x,ts) {
    return((x>=1)  && (x<=LIMIT) && (ts<=time.max))
  }
  
  # dt ~ normal distribution
  dt = rnorm(1, mean=miu, sd=sigma)
  
  while (within.range(last(x)+ u.x, last(ts)+ dt)) {
    # move in the direction specified by u.x and time difference dt
    x = append(x, last(x) + u.x)
    ts = append(ts, last(ts) + dt)
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
  while (last(noise.ts) < time.max) {
    noise.ts = append(noise.ts, last(noise.ts) + rexp(1,noise.freq))
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


stomach.sim.one2d = function(limit, time.max, ts, x, y, u.x, u.y, p, 
                             noise.freq, miux, miuy, sigma, ratio) {
  # function to test whether the spike is still inside the 1x8 grid 
  within.range = function(y,ts) {
    return((y>=1) && (y<=LIMIT) && (ts<=time.max))
  }
  
  all.lines = list("x.observ"=c(), "y.observ"=c(), "ts.observ"=c())
  
  for (i in 1:limit){
    for (j in 1:limit){
      #calculate the distantce from (i,j) to initial point (x,y)
      # distance = (i-x)^2+(j-y)^2
      # # e ~ normal distribution is the measurement error
      # e = rnorm(1, mean=0, sd=sigma)
      # #calculate the arrival time at point (i,j)
      # if (distance==0){
      # arrival.time = ts + e
      # } else{
      # arrival.time = ts + distance/(abs(i-x)*miux+abs(j-y)*miuy) + e
      # }
      distance = (i-x)^2+(j-y)^2
      # e ~ normal distribution is the measurement error
      e = rnorm(1, mean=0, sd=sigma)
      #calculate the arrival time at point (i,j)
      if (distance==0){
        arrival.time = ts + e
      } else{
        arrival.time = ts + sqrt(((i-x)/miux)^2 + ((j-y)/miuy)^2) + e
      }
      # only record signals that arrive before time.max
      if (arrival.time < time.max) {
        if (runif(1,0,1) > p){
          all.lines$x.observ = c(all.lines$x.observ, i)
          all.lines$y.observ = c(all.lines$y.observ, j)
          all.lines$ts.observ = c(all.lines$ts.observ, arrival.time)
        }
      }
    }
  }
  
  # while (within.range(y, ts)) {
  # one.line = stomach.sim.one(LIMIT, time.max, ts, 
  # x, u.x, p, noise.freq, miu, sigma)
  # all.lines$x.observ = c(all.lines$x.observ, one.line$x.observ)
  # all.lines$y.observ = c(all.lines$y.observ, rep(y, length(one.line$x.observ)))
  # all.lines$ts.observ = c(all.lines$ts.observ, one.line$ts.observ)
  # y = y + u.y
  # dt = rnorm(1, mean=ratio*miu, sd=ratio*sigma)
  # ts = ts + dt
  # }
  
  return(all.lines)
}


stomach.sim2d = function(limit, time.max, ts, x, y, u.x, u.y, p, noise.freq, miux, miuy, sigma,
                         gap, ratio) {
  SEED_BASE = 100;
  # SEED_BASE = 10000;
  set.seed(SEED_BASE * limit * time.max * ts * x * y * u.x * u.y * p * noise.freq *
             miux * miuy * sigma)
  
  signal.start.ts = ts
  all.lines2d = list("x.observ"=c(), "y.observ"=c(), "ts.observ"=c())
  while (signal.start.ts < time.max) {
    one.line2d = stomach.sim.one2d(limit, time.max, signal.start.ts, 
                                   x, y, u.x, u.y, p, noise.freq, miux, miuy, sigma, ratio)
    all.lines2d$x.observ = c(all.lines2d$x.observ, one.line2d$x.observ)
    all.lines2d$y.observ = c(all.lines2d$y.observ, one.line2d$y.observ)
    all.lines2d$ts.observ = c(all.lines2d$ts.observ, one.line2d$ts.observ)
    signal.start.ts = signal.start.ts + rexp(1, 1/gap)
  }
  
  
  # Generate noise with time ~ exponential distribution
  noise.ts = 0
  while (last(noise.ts) < time.max) {
    noise.ts = append(noise.ts, last(noise.ts) + rexp(1,noise.freq))
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
  num.expected.signal = abs(last(wave.position) - first(wave.position)) + 1
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
  
  green.indices = intersect(true.signal.indices, predict.signal.indices)
  blue.indices = intersect(true.noise.indices, predict.signal.indices)
  red.indices = intersect(true.signal.indices, predict.noise.indices)
  black.indices = intersect(true.noise.indices, predict.noise.indices)
  
  plot(ts[green.indices], x[green.indices], col="green",
       xlim=c(min(ts)-10, max(ts)+10))
  points(ts[blue.indices], x[blue.indices], col="blue")
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
           cbind(signal.guess$label, false.negative.label),
           ps=8)
}

cone.model = function(p, ts.grid) {
  x0 = p[[1]]; y0 = p[[2]]; v = p[[3]]; t0 = p[[4]]
#   LIMIT = 96
#   x.grid = array(1:LIMIT, c(LIMIT, LIMIT))
#   y.grid = t(x.grid)
  total = sum(
    (sqrt((x.grid - x0)^2 + (y.grid - y0)^2)/v - (ts.grid - t0))^2 *
      (ts.grid > 0)
  )
#   print(total)
  return(total)
}

cone.model.inverted = function(p, ts.grid) {
  x0 = p[[1]]; y0 = p[[2]]; v = p[[3]]; t0 = p[[4]]
#   LIMIT = 96
#   x.grid = array(1:LIMIT, c(LIMIT, LIMIT))
#   y.grid = t(x.grid)
  total = sum(
    (sqrt((x.grid - x0)^2 + (y.grid - y0)^2)*v - (ts.grid - t0))^2 *
      (ts.grid > 0)
  )
#   print(total)
  return(total)
}

cone.model.inverted.all = function(p, sim.matrix) {
  x0 = p[[1]]; y0 = p[[2]]; v = p[[3]]; t0 = p[[4]]
  #   LIMIT = 96
  #   x.grid = array(1:LIMIT, c(LIMIT, LIMIT))
  #   y.grid = t(x.grid)
  total = sum((sqrt((sim.matrix[,1] - x0)^2 + (sim.matrix[,2] - y0)^2)*v - 
            (sim.matrix[,3] - t0))^2) 
  #   print(total)
  return(total)
}

# cone.model.inverted.vt = function(p, ts.grid, x, y) {
#   pp = c(x,y, p[[1]], p[[2]]);
#   return(cone.model.inverted(pp, ts.grid))
# }

cone.model.inverted.xy = function(p, ts.grid, v, t) {
  pp = c(p[[1]], p[[2]], v, t);
  return(cone.model.inverted(pp, ts.grid))
}

cone.model.inverted.all.xy = function(p, sim.matrix, v, t) {
  pp = c(p[[1]], p[[2]], v, t);
  return(cone.model.inverted.all(pp, sim.matrix))
}

cone.model.square = function(p, ts.grid) {
  x0 = p[[1]]; y0 = p[[2]]; v = p[[3]]; t0 = p[[4]]
#   LIMIT = 96
#   x.grid = array(1:LIMIT, c(LIMIT, LIMIT))
#   y.grid = t(x.grid)
  total = sum(
    (((x.grid - x0)^2 + (y.grid - y0)^2)/(v^2) - ((ts.grid - t0)^2))^2 *
      (ts.grid > 0)
  )
#   print(total)
  # print(p)
  # return(log10(total))
  return(total)
}

convert.to.grid = function(sim.result, limit) {
  ts.grid = array(0, c(LIMIT, LIMIT))
  for (x in 1:limit){
    for (y in 1:limit){
      tss = sim.result$ts[which((sim.result$x == x)&(sim.result$y == y))]
      for (ts in tss) {
        ts.grid[x,y] = ts
      }
    }
  }
  return(ts.grid)
}


LIMIT = 96
x.grid = array(1:LIMIT, c(LIMIT, LIMIT))
y.grid = t(x.grid)

norm.xy = function(x0, y0){
  result = ((x.grid - x0)^2 + (y.grid - y0)^2)^(1/2)
  return (result)
}

dx0 = function(p, ts.grid){
  x0 = p[[1]]; y0 = p[[2]]; v = p[[3]]; t0 = p[[4]]
  total = sum(((1/v^2 - (norm.xy(x0, y0))^(-1)/ v *
                  (ts.grid - t0)) * (x.grid - x0)) * 
                (ts.grid > 0))
  result = (-2) * total
  return (result)
}

dy0 = function(p, ts.grid){
  x0 = p[[1]]; y0 = p[[2]]; v = p[[3]]; t0 = p[[4]]
  total = sum(((1/v^2 - (norm.xy(x0, y0))^(-1)/ v * 
                  (ts.grid - t0)) * (y.grid - y0)) * 
                (ts.grid > 0))
  result = (-2) * total
  return (result)
}

dv = function(p, ts.grid){
  x0 = p[[1]]; y0 = p[[2]]; v = p[[3]]; t0 = p[[4]]
  total = sum(((norm.xy(x0, y0))^2/v^3 - norm.xy(x0, y0)/v^2 * 
                 (ts.grid - t0)) * 
                (ts.grid > 0))
  result = (-2) * total
  return (result)
}

dt0 = function(p, ts.grid){
  x0 = p[[1]]; y0 = p[[2]]; v = p[[3]]; t0 = p[[4]]
  total = sum((norm.xy(x0, y0)/v - (ts.grid - t0)) * 
                (ts.grid > 0))
  result = (2) * total
  return (result)
}

dx0x0 = function(p, ts.grid){
  x0 = p[[1]]; y0 = p[[2]]; v = p[[3]]; t0 = p[[4]]
  total = sum(((x.grid - x0)^2*(ts.grid - t0)/v * 
                 (norm.xy(x0, y0))^(-3) + 1/v^2 - (ts.grid - t0)/v *
                 (norm.xy(x0, y0))^(-1)) * (ts.grid > 0))
  result = (2) * total
  return (result)
}

dy0y0 = function(p, ts.grid){
  x0 = p[[1]]; y0 = p[[2]]; v = p[[3]]; t0 = p[[4]]
  total = sum(((y.grid - y0)^2*(ts.grid - t0)/v * 
                 (norm.xy(x0, y0))^(-3) + 1/v^2 - (ts.grid - t0)/v *
                 (norm.xy(x0, y0))^(-1))* (ts.grid > 0))
  result = (2) * total
  return (result)
}

dvv = function(p, ts.grid){
  x0 = p[[1]]; y0 = p[[2]]; v = p[[3]]; t0 = p[[4]]
  total = sum((3*(norm.xy(x0, y0))^2/(v^4) - 2*norm.xy(x0, y0)/(v^3) * 
                 (ts.grid - t0)) * (ts.grid > 0))
  result = (2) * total
  return (result)
}

dt0t0 = function(p, ts.grid){
  x0 = p[[1]]; y0 = p[[2]]; v = p[[3]]; t0 = p[[4]]
  total = sum(ts.grid > 0)
  result = (2) * total
  return (result)
}

dx0y0 = function(p, ts.grid){
  x0 = p[[1]]; y0 = p[[2]]; v = p[[3]]; t0 = p[[4]]
  total = sum((x.grid-x0)*(y.grid-y0)*(ts.grid-t0)/v * 
                (norm.xy(x0,y0))^(-3) * (ts.grid > 0))
  result = (2) * total
  return (result)
}

dx0v = function(p, ts.grid){
  x0 = p[[1]]; y0 = p[[2]]; v = p[[3]]; t0 = p[[4]]
  total = sum((2/v^3 - (ts.grid-t0)/v^2 * (norm.xy(x0,y0))^(-1))*
                (x.grid-x0) * (ts.grid > 0))
  result = (2) * total
  return (result)
}

dx0t0 = function(p, ts.grid){
  x0 = p[[1]]; y0 = p[[2]]; v = p[[3]]; t0 = p[[4]]
  total = sum((x.grid-x0)/v * (norm.xy(x0,y0))^(-1) * (ts.grid > 0))
  result = (-2) * total
  return (result)
}

dy0v = function(p, ts.grid){
  x0 = p[[1]]; y0 = p[[2]]; v = p[[3]]; t0 = p[[4]]
  total = sum((2/v^3 - (ts.grid-t0)/v^2 * (norm.xy(x0,y0))^(-1))*
                (y.grid-y0) * (ts.grid > 0))
  result = (2) * total
  return (result)
}

dy0t0 = function(p, ts.grid){
  x0 = p[[1]]; y0 = p[[2]]; v = p[[3]]; t0 = p[[4]]
  total = sum((y.grid-y0)/v * (norm.xy(x0,y0))^(-1) * (ts.grid > 0))
  result = (-2) * total
  return (result)
}

dvt0 = function(p, ts.grid){
  x0 = p[[1]]; y0 = p[[2]]; v = p[[3]]; t0 = p[[4]]
  total = sum(norm.xy(x0,y0) / v^2 * (ts.grid > 0))
  result = (-2) * total
  return (result)
}

gra = function(p, ts.grid){
  a = c()
  a[1]=dx0(p,ts.grid)
  a[2]=dy0(p,ts.grid)
  a[3]=dv(p,ts.grid)
  a[4]=dt0(p,ts.grid)
  return(a)
}

hes = function(p, ts.grid){
  a = array(0, c(4,4))
  a[1,1] = dx0x0(p, ts.grid)
  a[1,2] = a[2,1] = dx0y0(p, ts.grid)
  a[1,3] = a[3,1] = dx0v(p, ts.grid)
  a[1,4] = a[4,1] = dx0t0(p, ts.grid)
  a[2,2] = dy0y0(p, ts.grid)
  a[2,3] = a[3,2] = dy0v(p, ts.grid)
  a[2,4] = a[4,2] = dy0t0(p, ts.grid)
  a[3,3] = dvv(p, ts.grid)
  a[3,4] = a[4,3] = dvt0(p, ts.grid)
  a[4,4] = dt0t0(p, ts.grid)
  return(a)
}

newton.cone = function(p, ts.grid) {
  iter.max = 200
  for (i in 1:iter.max) {
    hessian.cone = hes(p, ts.grid)
    gradient.cone = gra(p, ts.grid)
    p = p - solve(hessian.cone, gradient.cone)
    print(norm(as.matrix(gradient.cone), 'F'))
    print(cone.model(p,ts.grid))
    print(p)
    print(hessian.cone)
  }
  return(p)
}

normv = function(v) { norm(as.matrix(v), 'F') }

newton.cone.square = function(p, ts.grid) {
  iter.max = 200
  epsilon = 1e-6
  for (i in 1:iter.max) {
    print(p)
    p.old = p
    hessian.cone = hessian(cone.model.square, p, method="Richardson",
                           method.args=list(eps=1e-4, d=0.1, r=4, v=2),
                           ts.grid)
    gradient.cone = grad(cone.model.square, p, method="Richardson",
      method.args=list(eps=1e-4, d=0.0001, r=4, v=2, show.details=FALSE),
      ts.grid)
    p = p - solve(hessian.cone, gradient.cone)
    relerror = normv(p - p.old)/normv(p.old)
    if (relerror < epsilon) { break }
    # print(norm(as.matrix(gradient.cone), 'F'))
    # print(cone.model(p,ts.grid))
    # print(hessian.cone)
  }
  print(p)
  return(p)
}

grid.optim.find.best.point.1d = function(
    index, pp.current, num.splits, pp.lower, pp.upper, callback, ...) {
  num.spaces = num.splits - 1
  min.value = 1e100
  min.pp = pp.current
  for (i in 1:(num.spaces-1)) {
    pp.current[index] = pp.lower[index] + 
        (pp.upper[index] - pp.lower[index]) / num.spaces * i
    if (index < length(pp.current)) {
      pp.value.pair = grid.optim.find.best.point.1d(
                                  index + 1, pp.current, num.splits, 
                                  pp.lower, pp.upper, callback, ...)
    } else {
      pp.value.pair = list('value'=callback(pp.current, ...) , 'pp'=pp.current)
    }
    if (pp.value.pair$value < min.value) {
      min.value = pp.value.pair$value 
      min.pp = pp.value.pair$pp
    }
  }
  return(list('value'=min.value, 'pp'=min.pp))
}    

grid.optim.find.best.point = function(
    num.splits, pp.lower, pp.upper, callback, ...) {
  pp.current = pp.lower
  return(grid.optim.find.best.point.1d(
         1, pp.current, num.splits, pp.lower, pp.upper, callback, ...))
}

grid.optim = function(pp.lower.start, pp.upper.start, callback, ...) {
  num.splits = 4
  num.spaces = num.splits - 1
  iter.max = 1000
  epsilon = 1e-8
  pp.lower = pp.lower.start; pp.upper = pp.upper.start;
  for (iter in 1:iter.max) {
    pp.old = (pp.lower + pp.upper) / 2
    pp.value.pair = 
      grid.optim.find.best.point(num.splits, pp.lower, pp.upper, callback, ...)
    new.space = (pp.upper - pp.lower) / num.spaces
#     print(pp.value.pair$pp)
#     print(pp.value.pair$value)
    pp.lower = pp.value.pair$pp - new.space
    pp.upper = pp.value.pair$pp + new.space

    # reldiff = normv(pp.value.pair$pp - pp.old)/normv(pp.old)
    if (normv(new.space) < epsilon) { break }
  }
  return((pp.lower + pp.upper) / 2)
}

vt.optim = function(p, ts.grid){
  x0 = p[[1]]; y0 = p[[2]] 
  y = as.vector(ts.grid)
  x = as.vector(sqrt((x.grid - x0)^2 + (y.grid - y0)^2))
  x = x[y>0]; y = y[y>0]
  vt = lm(y ~ x)
  p[[4]] = vt$coefficients[[1]]
  p[[3]] = vt$coefficients[[2]]
#   print(vt$coefficients)
  return (p)
}

vt.optim.all = function(p, sim.matrix){
  x0 = p[[1]]; y0 = p[[2]] 
  y = sim.matrix[,3]
  x = as.vector(sqrt((sim.matrix[,1] - x0)^2 + (sim.matrix[,2] - y0)^2))
  vt = lm(y ~ x)
  p[[4]] = vt$coefficients[[1]]
  p[[3]] = vt$coefficients[[2]]
  #   print(vt$coefficients)
  return (p)
}

hybrid.optim = function(sim.matrix){
  pp.lower = c(-500,-500); pp.upper = c(500, 500);
  v = 1; t = 1
  iter.max = 10000
  epsilon = 1e-8
  pp.old = c(1,1,1,1)
  for (iter in 1:iter.max) {
    pp.estimate = grid.optim(pp.lower, pp.upper, cone.model.inverted.all.xy, sim.matrix,
                             v,t)
    pp.estimate = vt.optim.all(pp.estimate, sim.matrix)
#     print(pp.estimate)
#     print(cone.model.inverted.all(pp.estimate, sim.matrix))
    v = pp.estimate[[3]]
    t = pp.estimate[[4]]
    reldiff = normv(pp.estimate  - pp.old)/normv(pp.old)
    if (reldiff < epsilon) { break }
    pp.old = pp.estimate
  }
   return (pp.estimate)
 }

visualize2d = function(theta, plot.3d = FALSE) {
  limit=96
  # limit=4
  # time.max=limit
  time.max=limit 
  gap = 1000000
  lambda = theta$lambda; p = theta$p; sigma = theta$sigma;
  x = 48; y = 48; u.x = 1; u.y = 1; ts = 2; ratio = 10; miux = 1; miuy = miux;
  # x = 2; y = 2; u.x = 1; u.y = 1; ts = 1; ratio = 10;
  sim.result = stomach.sim2d(limit, time.max, ts, x, y, u.x, u.y, p, lambda, 
                             miux, miuy, sigma, gap, ratio)
  sim.matrix = as.matrix(sim.result)
  z.true = sim.result$z
  true.signal.indices = which(z.true %in% 1)
  true.noise.indices = which(z.true %in% 0)
  num.true.signal = length(true.signal.indices)
  num.true.noise = length(true.noise.indices)
#   print(sim.matrix)
#   ts.grid = convert.to.grid(sim.result, limit)
  pp.estimate = hybrid.optim(sim.matrix)
  mean.travelling.time = mean(sim.result$ts[true.signal.indices])
  pp.estimate = append(pp.estimate, c(num.true.signal, num.true.noise,
                       mean.travelling.time))

  if (plot.3d) {
    z.true = sim.result$z
    true.signal.indices = which(z.true %in% 1)
    true.noise.indices = which(z.true %in% 0)
    plot3d(sim.result$x[true.noise.indices], sim.result$y[true.noise.indices], 
           sim.result$ts[true.noise.indices], col="black", size=3, 
           xlab="x", ylab="y", zlab="t"
           #          aspect=c(limit, limit, time.max)
    )
    points3d(sim.result$x[true.signal.indices], sim.result$y[true.signal.indices], 
             sim.result$ts[true.signal.indices], col="green", size=3)
    rgl.snapshot("plots/cone_simulate.png")
    # plot3d(sim.result$x[true.signal.indices], sim.result$y[true.signal.indices], 
           # sim.result$ts[true.signal.indices], col="green", size=3)
    #   z.true = sim.result$z
    #   x = sim.result$x
    #   ts = sim.result$ts
    #   observ = sim.result[,1:2]
    #   
    #   hough.result = hough(ts, x)
    #   all.indices = 1:length(x)
    #   signal.guess = hough.result$signal.guess
    #   predict.signal.indices = unlist(signal.guess$indices)
    #   predict.noise.indices = setdiff(all.indices, predict.signal.indices)
    #   true.signal.indices = which(z.true %in% 1)
    #   true.noise.indices = which(z.true %in% 0)
    #   
    #   green.indices = intersect(true.signal.indices, predict.signal.indices)
    #   blue.indices = intersect(true.noise.indices, predict.signal.indices)
    #   red.indices = intersect(true.signal.indices, predict.noise.indices)
    #   black.indices = intersect(true.noise.indices, predict.noise.indices)
    #   
    #   plot(ts[green.indices], x[green.indices], col="green",
    #        xlim=c(min(ts)-10, max(ts)+10))
    #   points(ts[blue.indices], x[blue.indices], col="blue")
    #   points(ts[red.indices], x[red.indices], col="red")
    #   points(ts[black.indices], x[black.indices], col="black")
    #   false.negative.label = c()
    #   if (length(red.indices) > 0) {
    #     for (i in 1:length(red.indices)) {
    #       data.index = red.indices[i]
    #       line = point.angle.to.line(ts[data.index],
    #                                  x[data.index],
    #                                  hough.result$best.angle)
    #       false.negative.label[i] = line$debug.string
    #     }
    #   }
    #   identify(ts[cbind(predict.signal.indices, red.indices)], 
    #            x[cbind(predict.signal.indices, red.indices)], 
    #            cbind(signal.guess$label, false.negative.label),
    #            ps=8)
  }
  return(list('truth'=c(x,y,miux,ts), 'pp'=pp.estimate, 
              'mean.travelling.time'=mean.travelling.time))
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

hough = function(xs, ys) {
  num.data = length(xs)
  angle.distance.to.points = hash()
  for (i in 1:num.data) {
    x = xs[[i]]
    y = ys[[i]]
    # Discretization at 10 degree intervals
    for (angle in (1:36)*5) {
      line = point.angle.to.line(x,y,angle)
      if (!has.key(line$key, angle.distance.to.points)) {
        angle.distance.to.points[[line$key]] = c()
      }
      angle.distance.to.points[[line$key]] = 
        rbind(angle.distance.to.points[[line$key]], 
              list("indices"=i, "label"=line$debug.string))
    }
  }
  
  num.points.threshold = 15
  indices.above.threshold = c()
  best.angle = -1
  for (key in keys(angle.distance.to.points)) {
    point.indices = angle.distance.to.points[[key]]
    num.points = nrow(point.indices)
    if (num.points > num.points.threshold) {
      print(num.points)
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

relerror.plot = function(pp.estimates, truth, label, variables, plot.value, 
                         should.log, should.use.snr, params, add.text, 
                         should.use.travel = FALSE){
  snr = c(); 
  relerror.x = c(); relerror.y = c(); relerror.v = c(); relerror.t = c()
  for (i in 1:nrow(pp.estimates)){
    snr = append(snr, pp.estimates[i,6]/pp.estimates[i,7])
  }
  if (plot.value){
    relerror.x = pp.estimates[,2]
    relerror.y = pp.estimates[,3]
    relerror.v = 1/pp.estimates[,4]
    relerror.t = pp.estimates[,5]
  }else{
    for (i in 1:nrow(pp.estimates)){
      relerror.x = append(relerror.x, abs(pp.estimates[i,2]-truth[[1]])/truth[[1]])
      relerror.y = append(relerror.y, abs(pp.estimates[i,3]-truth[[2]])/truth[[2]])
      relerror.v = append(relerror.v, abs(1/pp.estimates[i,4]-truth[[3]])/truth[[3]])
      relerror.t = append(relerror.t, abs(pp.estimates[i,5]-truth[[4]])/truth[[4]])
    }
  }
  
#   print(snr)
#   print(relerror.x)
#   print(relerror.y)
#   print(relerror.v)
#   print(relerror.t)
  # plot(log10(snr), log10(relerror.x), type = "p", main = "Relative Error of X0")
    log10snr = snr
    log10.relerror.x = relerror.x
    log10.relerror.y = relerror.y
    log10.relerror.v = relerror.v
    log10.relerror.t = relerror.t
  legend.position = "topright"
  if (!should.use.snr) {
    log10snr = pp.estimates[,1]
    legend.position = "topleft"
  }
  if (should.use.travel) {
    log10snr = pp.estimates[,1]/pp.estimates[,8]
  }
  
  shape = 21:24
  ys = cbind(log10.relerror.x, log10.relerror.y, log10.relerror.v, log10.relerror.t)
  
  xmin = min(log10snr); xmax = max(log10snr)
  if (variables == 4){
    ymin = min(ys[,variables]);
    print (ymin)
    ymax = max(ys[,variables]);
    print(ymax)
  } else {
    ymin = min(log10.relerror.x, log10.relerror.y,
               log10.relerror.v, log10.relerror.t);
    ymax = max(log10.relerror.x, log10.relerror.y,
               log10.relerror.v, log10.relerror.t) * 1.1;
  }
  
  dot.colors = c("red", "blue", "green", "purple")
  dot.names = c(expression("x"[0]), expression("y"[0]), 
                expression("v"), expression("t"[0]))
  
  i = variables[[1]]
  ymin = min(ys[,i], truth[[i]])
  ymax = max(ys[,i], truth[[i]])
  diff = ymax - ymin
  ymin = ymin - diff *0.1
  ymax = ymax + diff *0.1
  if (should.log) {
    plot(c(),c(),xlim=c(xmin, xmax), ylim=c(ymin, ymax), 
         xlab=label[[1]], ylab=label[[2]], log="x")
  } else {
    plot(c(),c(),xlim=c(xmin, xmax), ylim=c(ymin, ymax), 
         xlab=label[[1]], ylab=label[[2]])
  }
  params = round(params, digits = 2)
  # for (i in variables) {
    # lines(log10snr, ys[,i], type = "p", col=dot.colors[[i]], shape[[i]]);
    # text(log10snr, ys[,i], params, cex=1, pos=3, col="red")
    # if (plot.value) {
      # lines(log10snr, rep(truth[[i]], length(log10snr)), 
            # type = "l", col=dot.colors[[i]])
    # }
  # }
  print(log10snr)
  lines(log10snr, ys[,i], type = "p", col="red", shape[[i]]);
  if (add.text) {
    text(log10snr, ys[,i], params, cex=1, pos=3, col="red")
  }
  if (plot.value) {
    lines(log10snr, rep(truth[[i]], length(log10snr)), 
          type = "l", col="green")
  }
  # legend(legend.position, dot.names, col=dot.colors, pch=shape)
}

optimality.table = function() {
  pp.estimates = c()
  plot.mode = 3
  variables.label = c(expression('x'[0]), expression('y'[0]), 
                      'v', expression('t'[0]))
                      
  if (plot.mode==1) {
    lambdas = 2^((-6:16)/2)
    # lambdas = 2^(c(-6, -5, -4, 16)/2)
    for (lambda in lambdas){
      theta.true = list("lambda"=lambda, "p"=0, "sigma"=0.000001)
      truth.estimate.pair = visualize2d(theta.true)
      pp.estimates = rbind(pp.estimates, truth.estimate.pair$pp)
      print(lambda)
    }
    pp.estimates = cbind(lambdas, pp.estimates)
    for (variable in 1:4) {
      relerror.plot(pp.estimates, truth.estimate.pair$truth, 
                    c("Signal to noise ratio", variables.label[[variable]]),
                    variable, TRUE, TRUE, TRUE, lambdas, TRUE)
      dev.copy(pdf,paste('plots/plot_mode', plot.mode, 
                         '_',variable ,'.pdf', sep=''))
      dev.off()
    }
  }
  
  if (plot.mode==2){
    ps = (0:8)/10
    for (p in ps){
      theta.true = list("lambda"=1, "p"=p, "sigma"=0.000001)
      truth.estimate.pair = visualize2d(theta.true)
      pp.estimates = rbind(pp.estimates, truth.estimate.pair$pp)
      print(p)
    }
    pp.estimates = cbind(ps, pp.estimates)
    for (variable in 1:4) {
      relerror.plot(pp.estimates, truth.estimate.pair$truth, 
                    c("p", variables.label[[variable]]),
                    variable, TRUE, FALSE, FALSE, pp.estimates[,1],FALSE)
      dev.copy(pdf,paste('plots/plot_mode', plot.mode, 
                         '_',variable ,'.pdf', sep=''))
      dev.off()
    }
  }
  
  if (plot.mode==3){
    sigmas = (1:10)/10
    # sigmas = c(1,10)
    # sigmas = (1:10)
    for (sigma in sigmas){
      theta.true = list("lambda"=0.000001, "p"=0, "sigma"=sigma)
      truth.estimate.pair = visualize2d(theta.true)
      pp.estimates = rbind(pp.estimates, truth.estimate.pair$pp)
      print(sigma)
    }
    pp.estimates = cbind(sigmas, pp.estimates)
    for (variable in 1:4) {
      relerror.plot(pp.estimates, truth.estimate.pair$truth, 
                    c('Ratio of measurement error to mean travelling time', 
                    variables.label[[variable]]),
                    variable, TRUE, FALSE, FALSE, pp.estimates[,1],TRUE, TRUE)
      dev.copy(pdf,paste('plots/plot_mode', plot.mode, 
                         '_',variable ,'.pdf', sep=''))
      dev.off()
    }
  }

  if (plot.mode==4) {
    # sigmas = (1:10)/10
    sigmas = 10
    for (sigma in sigmas){
      theta.true = list("lambda"=100, "p"=0, "sigma"=0.000001)
      truth.estimate.pair = visualize2d(theta.true, TRUE)
      pp.estimates = rbind(pp.estimates, truth.estimate.pair$pp)
    }
  }

  if (plot.mode==5) {
    # sigmas = (1:10)/10
    theta.true = list("lambda"=1, "p"=0.1, "sigma"=1)
    truth.estimate.pair = visualize2d(theta.true, TRUE)
  }

  
  if (plot.mode==6){
    ps = (0:8)/10
    for (p in ps){
      theta.true = list("lambda"=0.0000001, "p"=p, "sigma"=0.000001)
      truth.estimate.pair = visualize2d(theta.true)
      pp.estimates = rbind(pp.estimates, truth.estimate.pair$pp)
      print(p)
    }
    pp.estimates = cbind(ps, pp.estimates)
    print(pp.estimates)
    relerror.plot(pp.estimates, truth.estimate.pair$truth, 
                  c("p", "relerror"), 4, TRUE, FALSE, FALSE)
  }
  
  if (plot.mode==7){
    ps = (0:8)/10
    for (p in ps){
      theta.true = list("lambda"=1, "p"=p, "sigma"=0.000001)
      truth.estimate.pair = visualize2d(theta.true)
      pp.estimates = rbind(pp.estimates, truth.estimate.pair$pp)
      print(p)
    }
    pp.estimates = cbind(ps, pp.estimates)
    #     print(pp.estimates)
    relerror.plot(pp.estimates, truth.estimate.pair$truth, 
                  c("snr", "estimate.t0"), 4, TRUE, FALSE, TRUE)
  }

  if (plot.mode==8){
    lambdas = 1
    for (lambda in lambdas){
      theta.true = list("lambda"=lambda, "p"=0, "sigma"=0.000001)
      truth.estimate.pair = visualize2d(theta.true, TRUE)
    }
  }
  
  
#   print(pp.estimates)
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


