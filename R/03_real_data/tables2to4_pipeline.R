options(error=recover)
require(hash)
require(rgl)
# install.packages("signal")
require(signal)
require(numDeriv)

run.smoothing = TRUE
plot.first.channel = FALSE
LOWESS = 1; BUTTER = 2; MIXED = 3
# smooth.method = BUTTER
# smooth.method = LOWESS
smooth.method = BUTTER
run_hough = TRUE
run.hybrid.optim = TRUE
run.plane.model = TRUE

if (run.smoothing) {
  # some comment
  # filename = "C:/Users/Amy Lu/Documents/My Dropbox/Research/Simulate/CM_PIN_Control_2N30_Aunor-txt.csv"
  # filename = "../Data_CK/heart/CM_PIN_Control_2N30_Aunor_2_input.csv"
  # filename = "../Data_CK/heart/CM_PIN_Control_2N30_Aunor_2_input_60000.csv"
  filename = "Data_CK/heart/CM_PIN_Control_2N30_Aunor_2_input_60000.csv"
  data = read.csv(filename)
  data = data.matrix(data)
  l = ncol(data);
  f = function(str) {as.integer(substr(str,3,4))};
  channel = sapply(colnames(data)[2:l], f)
  
  time = data[,1]
  # activation.time = c()
  # all.lines = list("channel"=c(), "min.index"=c(), "ts.observ"=c())
  all.lines = list("x.observ"=c(), "y.observ"=c(), "ts.observ"=c())
  min.index = c()
  
  # time.start = proc.time()[1]
  for (i in 2:ncol(data)){
       # i =10
    unsmooth=data[,i]
    if (smooth.method == LOWESS) {
      smooth = lowess(1:length(unsmooth),unsmooth, 
                      f = 20/length(unsmooth),delta=1/length(unsmooth))
      min.index.gap = 2000
    } else if (smooth.method == MIXED) {
      smooth = lowess(1:length(unsmooth),unsmooth, 
                      f = 20/length(unsmooth),delta=1/length(unsmooth))
      butter.filter = butter(2, 1/500, type="high")
      smooth = filter(butter.filter, smooth$y)
      smooth$y = smooth
      smooth$x = 1:length(smooth$y)
      lower.threshold = quantile(smooth$y, 0.0005)
      min.index.gap = 500
    } else if (smooth.method == BUTTER) {
      # butter.filter = butter(2, 1/50, type="high")
      butter.filter = butter(2, 1/500, type="high")
      smooth = filter(butter.filter, unsmooth)
      smooth$y = smooth
      smooth$x = 1:length(smooth$y)
      lower.threshold = quantile(smooth$y, 0.0005)
      min.index.gap = 500
    }
    change.smooth = smooth
    smooth.y.mean = mean(smooth$y) 
    smooth.y.sd = sd(smooth$y)
    this.min.index = c()
    while (1) {
      if (smooth.method == LOWESS) {
        if (smooth$y[which.min(change.smooth$y)] >=
            (smooth.y.mean - 2.5 * smooth.y.sd )) {
          break
        }
      } else if (smooth.method == BUTTER) {
        if (smooth$y[which.min(change.smooth$y)] >= lower.threshold) {
          break
        }
      } else if (smooth.method == MIXED) {
        if (smooth$y[which.min(change.smooth$y)] >= lower.threshold) {
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
      
      min.index = which.min(change.smooth$y)
      this.min.index = c(this.min.index, min.index)
      
      ts.observ = time[min.index]
      all.lines$ts.observ = c(all.lines$ts.observ, ts.observ)
      
      lower.bound = max(1, min.index-min.index.gap)
      upper.bound = min(length(change.smooth$y), min.index+min.index.gap)
      change.smooth$y[lower.bound:upper.bound] = 1000000
    }
    
    if (plot.first.channel) {
      print('here')
      plot(data[,i], col="green")
      lines(smooth,col=4)
      #       points(min.index, smooth$y[min.index], col="red")
      points(this.min.index, smooth$y[this.min.index], col="red")
      break
    }
  }
  # time.end= proc.time()[1]
  # print('time used')
  # print(time.end-time.start)
  all.lines = as.data.frame(all.lines)
  all.lines = all.lines[order(all.lines$ts.observ),]
  all.lines$ts.observ= all.lines$ts.observ/200
  
}
# print(all.lines)
# print(sort(table(all.lines$channel)))
# sim.matrix = as.matrix(all.lines)

stomach.plot3d = function(all.result) {
  
  #   green.indices = which(all.result$z == 1 & all.result$prediction == 1)
  #   yellow.indices = which(all.result$z == 0 & all.result$prediction == 1)
  #   red.indices = which(all.result$z == 1 & all.result$prediction == 0)
  #   black.indices = which(all.result$z == 0 & all.result$prediction == 0)
  #   
  ts = all.result$ts.observ; x = all.result$x.observ; y = all.result$y.observ;
  
  #   plot3d(x[black.indices], y[black.indices], ts[black.indices], 
  #          # points3d(ts[black.indices], x[black.indices], y[black.indices], 
  #          col="black", size=3,
  #          xlim=c(min(x), max(x)),
  #          ylim=c(min(y), max(y)),
  #          zlim=c(min(ts), max(ts)))
  
  plot3d(x, y, ts,
         # points3d(ts[black.indices], x[black.indices], y[black.indices], 
         col="black", size=3,
         xlim=c(min(x), max(x)),
         ylim=c(min(y), max(y)),
         zlim=c(min(ts), max(ts)))
  
  #   while (1) {
  #     is.selected = select3d(button = c("right"))
  #     keys = all.result$keys
  #     indices = c()
  #     for (i in 1:length(ts)) {
  #       xi = x[[i]]; yi = y[[i]]; tsi = ts[[i]]
  #       if (is.selected(x[[i]], y[[i]], ts[[i]])) {
  #         indices = c(indices, i)
  #       }
  #     }
  #     x.zoom = x; y.zoom = y; ts.zoom = ts;
  #     if (length(indices) > 0) {
  #       x.zoom = x[indices]
  #       y.zoom = y[indices]
  #       ts.zoom = ts[indices]
  #     }
  #     plot3d(x.zoom, y.zoom, ts.zoom,
  #            col="black", size=3)
  #     # ,
  #     # xlim=c(min(x), max(x)),
  #     # ylim=c(min(y), max(y)),
  #     # zlim=c(min(ts), max(ts)))
  #   }
  
  #   points3d(x[yellow.indices], y[yellow.indices], ts[yellow.indices], 
  #            col="darkgoldenrod3", size=3)
  #   points3d(x[red.indices], y[red.indices], ts[red.indices], 
  #            col="red", size=3)
  #   points3d(x[green.indices], y[green.indices], ts[green.indices], 
  #            col="green", size=3)
}
# stomach.plot3d(all.lines)

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
  rho.step = get.step.key(rho, 0.05)
  phi.step = get.step.key(phi/pi*180, 2)
  theta.step = get.step.key(theta/pi*180, 2)
  key = paste(rho.step, phi.step, theta.step, sep=",")
  return(key)
}

is.point.on.plane = function(point, normal, rho) {
  distance.to.plane = abs(dot.product(point, normal) - rho)
#   return(distance.to.plane < 0.05)
  return(distance.to.plane < 0.1)
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

mode = function(vector.of.integers) {
  return(as.integer(names(sort(-table(vector.of.integers))[1])))
}

# hough.plane = function(data.frame.points) {
# data.points = as.matrix(data.frame.points)
hough.plane = function(data.points) {
  set.seed(1)
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
    # if ((n[[1]] == 0 & n[[2]] == 0) |
    # (n[[2]] == 0 & n[[3]] == 0) |
    if  ((n[[2]] == 0 & n[[3]] == 0) |
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
      # TODO: change the number 40 to depend on limit. e.g. number of channels*66%
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
#   print(max.normal.vector)
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
    plane.indices[indices.on.plane] = i
  }
  
  # find the unclassified points that can spanned a "true" plane  
    small.plane.threshold = 30
  #   i = 1
    plane.index = max(plane.indices)
    unclassify.indices = which(prediction == 0)
    while (length(unclassify.indices) >= 3) {
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
        plane.index = plane.index + 1
        prediction[indices.on.plane] = 1
        keys[indices.on.plane] = key
        n1[indices.on.plane] = max.normal.vector[[1]]
        n2[indices.on.plane] = max.normal.vector[[2]]
        n3[indices.on.plane] = max.normal.vector[[3]]
        rhos[indices.on.plane] = rho.signed
        plane.indices[indices.on.plane] = plane.index
      } else { 
        prediction[unclassify.indices[i]] = 2 
      }
      unclassify.indices = which(prediction == 0)
    }
    noise.indices = which(prediction == 2)
    prediction[noise.indices] = 0
  
  return(data.frame(prediction, keys, n1, n2, n3, plane.indices, rhos))
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

stomach.plot2d.interactive = function(all.result) {
  x = all.result$x; y = all.result$y; ts = all.result$ts; 
  n1 = all.result$n1; n2 = all.result$n2; n3 = all.result$n3; 
  rhos = all.result$rhos;
  stomach.plot3d(all.result)
  # code for drawing just one plane
  # for (i in 1:length(ts)) {
  # rho = rhos[[i]]
  # if (rho != 0) {
  # n = c(n1[[i]], n2[[i]], n3[[i]]); 
  # print(n)
  # print(rho)
  # draw.plane(n, rho)
  # draw.plane(n, rho+0.05)
  # draw.plane(n, rho-0.05)
  # break
  # }
  # }
  rgl.viewpoint( theta = 250, phi = 10, fov = 40)
  while (1) {
    #     is.selected = select3d(button = c("left"))
    is.selected = select3d(button = c("right"))
    keys = all.result$keys
    for (i in 1:length(ts)) {
      xi = x[[i]]; yi = y[[i]]; tsi = ts[[i]]
      if (is.selected(x[[i]], y[[i]], ts[[i]])) {
        key = keys[[i]]
        n = c(n1[[i]], n2[[i]], n3[[i]]); rho = rhos[[i]]
        draw.plane(n, rho*200)
        # draw.plane(n, rho+0.05)
        # draw.plane(n, rho-0.05)
        print(all.result[i,])
      }
    }
    indices.on.same.plane = which(keys == key)
    points3d(x[indices.on.same.plane], 
             y[indices.on.same.plane],
             ts[indices.on.same.plane],
             col="violet", size=9)
  }
}

if (run_hough) {
  hough.result = hough.plane(all.lines)
  all.result = cbind(all.lines, hough.result)
  # all.result = all.result[which(all.result$plane.indices == 1),]
  # all.result = all.lines
  all.result.plot = all.result
  all.result.plot$ts.observ = all.result.plot$ts.observ * 200
  # stomach.plot2d.interactive(all.result.plot)
  
  num.plane = max(all.result$plane.indices)
  #create an array to save the ts of each wave propagation at each channel
  result.array = array(-1, dim=c(8,8,num.plane))
  for (i in 1:nrow(all.result)){
    x = all.result$x.observ[i]
    y = all.result$y.observ[i]
    plane.index = all.result$plane.indices[i]
    ts = all.result$ts.observ[i] * 200
    if (plane.index==0){
      next
    }
    result.array[x, y, plane.index]=ts
  }
}

  LIMIT = 8
  x.grid = array(1:LIMIT, c(LIMIT, LIMIT))
  y.grid = t(x.grid)

cone.model.inverted = function(p, ts.grid) {
  x0 = p[[1]]; y0 = p[[2]]; v = p[[3]]; t0 = p[[4]]
  total = sum(
    (sqrt((x.grid - x0)^2 + (y.grid - y0)^2)*v - (ts.grid - t0))^2 *
      (ts.grid > 0)
  )
  #   print(total)
  return(total)
}

cone.model.inverted.all = function(p, sim.matrix) {
  x0 = p[[1]]; y0 = p[[2]]; v = p[[3]]; t0 = p[[4]]
  total = sum((sqrt((sim.matrix[,1] - x0)^2 + (sim.matrix[,2] - y0)^2)*v - 
                 (sim.matrix[,3] - t0))^2*(sim.matrix[,3]>0)) 
  #   print(total)
  return(total)
}

cone.model.inverted.xy = function(p, ts.grid, v, t) {
  pp = c(p[[1]], p[[2]], v, t);
  return(cone.model.inverted(pp, ts.grid))
}

cone.model.inverted.all.xy = function(p, sim.matrix, v, t) {
  pp = c(p[[1]], p[[2]], v, t);
  return(cone.model.inverted.all(pp, sim.matrix))
}

cone.model.inverted.all.xy2 = function(x,y, sim.matrix, v, t) {
  values = c()
  for (i in 1:length(x)) {
    pp = c(x[[i]], y[[i]], v, t);
    values = c(values, cone.model.inverted.all(pp, sim.matrix))
  }
  return(values)
}


grid.optim.find.best.point.1d = function(
  index, pp.current, num.splits, pp.lower, pp.upper, callback, ...) {
  num.spaces = num.splits - 1
  min.value = 1e100
  min.pp = pp.current
  # for (i in 1:(num.spaces-1)) {
  for (i in 0:num.spaces) {
    pp.current[index] = pp.lower[index] + 
      (pp.upper[index] - pp.lower[index]) / num.spaces * i
    if (index < length(pp.current)) {
      pp.value.pair = grid.optim.find.best.point.1d(
        index + 1, pp.current, num.splits, 
        pp.lower, pp.upper, callback, ...)
    } else {
      pp.value.pair = list('value'=callback(pp.current, ...) , 'pp'=pp.current)
      # print(c(pp.value.pair$value, pp.value.pair$pp, 
            # log(100/(pp.upper-pp.lower))/log(1.5)))
#       contains = (pp.lower[[1]] < -49.59845) && (pp.upper[[1]] > -49.59845) 
      # print(c(pp.value.pair$value, pp.lower, pp.upper, contains,
            # log(100/(pp.upper-pp.lower))/log(1.5)))
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
    for (i in 1:length(pp.value.pair$pp)) {
      if (abs(pp.value.pair$pp[[i]] - pp.lower[[i]]) < 1e-12) {
        pp.value.pair$pp[[i]] = pp.lower[[i]] + new.space[[i]]
      }
      if (abs(pp.value.pair$pp[[i]] - pp.upper[[i]]) < 1e-12) {
        pp.value.pair$pp[[i]] = pp.upper[[i]] - new.space[[i]]
      }
    }
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
  x = x[y>0]; y = y[y>0]
  vt = lm(y ~ x)
  p[[4]] = vt$coefficients[[1]]
  p[[3]] = vt$coefficients[[2]]
  #   print(vt$coefficients)
  return (p)
}

hybrid.optim = function(sim.matrix){
  pp.lower = c(-50,-50); pp.upper = c(50, 50);
  v = 1; t = 1
  iter.max = 10000
  epsilon = 1e-6
  pp.old = c(1,1,1,1)
  for (iter in 1:iter.max) {
    pp.estimate = grid.optim(pp.lower, pp.upper, cone.model.inverted.all.xy, 
                              sim.matrix, v,t)
    # pp.estimate = grid.optim(pp.lower, pp.upper, cone.model.inverted.xy, sim.matrix,
                             # v,t)
    pp.estimate = vt.optim.all(pp.estimate, sim.matrix)
    # pp.estimate = vt.optim(pp.estimate, sim.matrix)
    # print(c(pp.estimate, cone.model.inverted.all(pp.estimate, sim.matrix)))
    #     print(cone.model.inverted.all(pp.estimate, sim.matrix))
    v = pp.estimate[[3]]
    t = pp.estimate[[4]]
    reldiff = normv(pp.estimate  - pp.old)/normv(pp.old)
    if (reldiff < epsilon) { break }
    pp.old = pp.estimate
  }
  # grad.at.estimate = grad(cone.model.inverted.all, pp.estimate, 
                          # method="Richardson", method.args=list(), sim.matrix)
  # print('grad.at.estimate')
  # print(grad.at.estimate)
  # print('last run')
  # pp.estimate = grid.optim(pp.lower, pp.upper, cone.model.inverted.all.xy, 
                              # sim.matrix, v,t)
  # pp.estimate = vt.optim.all(pp.estimate, sim.matrix)

  # pp.lower = c(-50, -50); pp.upper = c(-49.48618,-49.48618)
  # xs =seq(pp.lower[[1]],pp.upper[[1]],0.01)
  # ys =seq(pp.lower[[2]],pp.upper[[2]],0.01)
  # zs = outer(xs,ys, FUN=cone.model.inverted.all.xy2, sim.matrix, v, t)
  # # mygrid <- expand.grid(x=seq(-49.9,-49.7, by=0.1), y=seq(-49.9,-49.7, by=0.1)) 
  # mygrid <- expand.grid(
          # x=seq(pp.lower[[1]],pp.upper[[1]],(pp.upper[[1]]-pp.lower[[1]])/(4-1)),
          # y=seq(pp.lower[[2]],pp.upper[[2]],(pp.upper[[2]]-pp.lower[[2]])/(4-1)))
  # filled.contour(xs,ys,zs, color.palette=rainbow, nlevels = 100, 
    # plot.axes={ axis(1); axis(2); points(mygrid, cex=0.5, pch=19)})
  # if (NA>0) {}

  return(pp.estimate)
}

normv = function(v) { norm(as.matrix(v), 'F') }

compute.loss = function(loss.fn, coef, data) {
  ssres = loss.fn(coef, data)
  ts = data[,3]
  ts.mean = mean(ts)
  sstot = sum((ts - ts.mean)^2)
  return(1-ssres/sstot)
}

print('cone model')
if (run.hybrid.optim) {
  planes = 1:num.plane
  for (i in planes) {
    x.observ = as.vector(x.grid)
    y.observ = as.vector(y.grid)
    ts.observ = as.vector(result.array[,,i])
    sim.matrix = cbind(x.observ, y.observ, ts.observ)
    sim.matrix = sim.matrix[which(ts.observ > 0),]
    pp.estimate = hybrid.optim(sim.matrix)
    print(list('pp'=pp.estimate, 'v'=1/pp.estimate[[3]]))
    grad.at.estimate = grad(cone.model.inverted.all, pp.estimate, 
                          method="Richardson", method.args=list(), sim.matrix)
    print('grad.at.estimate')
    print(grad.at.estimate)
    print('coefficient')
    coefficient = compute.loss(cone.model.inverted.all, pp.estimate, sim.matrix)
    print(coefficient)
  }
}

plane.model = function(p, sim.matrix) {
  a = p[[1]]; b = p[[2]]; c = p[[3]];
  total = sum((a*sim.matrix[,1] + b*sim.matrix[,2] + c - sim.matrix[,3])^2 * 
                (sim.matrix[,3]>0))
  return(total)
}

if (run.plane.model) {
  planes = 1:num.plane
  for (i in planes) {
    x.observ = as.vector(x.grid)
    y.observ = as.vector(y.grid)
    ts.observ = as.vector(result.array[,,i])
    sim.matrix = cbind(x.observ, y.observ, ts.observ)
    sim.matrix = sim.matrix[which(ts.observ > 0),]
    x.observ = sim.matrix[,1]
    y.observ = sim.matrix[,2]
    ts.observ = sim.matrix[,3]
    # nlm.estimate = nlm(plane.model, c(1,1,1))
    # estimate = nlm.estimate$estimate
    # a = estimate[[1]]; b = estimate[[2]]
    # v = 1/sqrt(a^2+b^2)
    # print('nlm_result')
    # print(c(a,b,v))

    vt = lm(ts.observ ~ x.observ + y.observ)
    a = vt$coefficients[[2]]; b = vt$coefficients[[3]];
    v = 1/sqrt(a^2+b^2)
    print('lm_result')
    print(c(a,b,v, ts.observ[[1]]))
    print('coefficient')
    params = c(vt$coefficients[[2]], vt$coefficients[[3]], vt$coefficients[[1]])
    coefficient = compute.loss(plane.model, params, sim.matrix)
    print(coefficient)
  }
}
