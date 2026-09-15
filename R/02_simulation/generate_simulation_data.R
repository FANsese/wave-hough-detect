# 生成胃电慢波模拟数据并保存为CSV
options(error = recover)
require(hash)

# 从原始文件复用的函数
last.element = function(x) { return(tail(x,1)) }

stomach.sim2d = function(limit, time.max, ts, x, y, u.x, u.y, p, noise.freq, miu, sigma,
                         gap, ratio, seed.shift = 1) {
  SEED_BASE = 101
  set.seed(SEED_BASE * limit * time.max * ts * x * y * u.x * u.y * p * noise.freq *
             miu * sigma + seed.shift)
  
  # 内部函数 - 生成单条信号线
  stomach.sim.one = function(LIMIT, time.max, ts, x, u.x, p, noise.freq, miu, sigma) {
    within.range = function(x,ts) {
      return((x>=1)  && (x<=LIMIT) && (ts<=time.max))
    }
    
    dt = rnorm(1, mean=miu, sd=sigma)
    while (within.range(last.element(x)+ u.x, last.element(ts)+ dt)) {
      x = append(x, last.element(x) + u.x)
      ts = append(ts, last.element(ts) + dt)
      dt = rnorm(1, mean=miu, sd=sigma)
    }
    
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
  
  # 内部函数 - 生成2D信号
  stomach.sim.one2d = function(LIMIT, time.max, ts, x, y, u.x, u.y, p, noise.freq, miu, sigma, ratio) {
    within.range = function(y,ts) {
      return((y>=1) && (y<=LIMIT) && (ts<=time.max))
    }
    
    all.lines = list("x.observ"=c(), "y.observ"=c(), "ts.observ"=c())
    while (within.range(y, ts)) {
      one.line = stomach.sim.one(LIMIT, time.max, ts, x, u.x, p, noise.freq, miu, sigma)
      all.lines$x.observ = c(all.lines$x.observ, one.line$x.observ)
      all.lines$y.observ = c(all.lines$y.observ, rep(y, length(one.line$x.observ)))
      all.lines$ts.observ = c(all.lines$ts.observ, one.line$ts.observ)
      y = y + u.y
      dt = rnorm(1, mean=ratio*miu, sd=ratio*sigma)
      ts = ts + dt
    }
    return(all.lines)
  }
  
  # 主生成逻辑
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
  
  # 生成噪声
  noise.ts = 0
  while (last.element(noise.ts) < time.max) {
    noise.ts = append(noise.ts, last.element(noise.ts) + rexp(1,noise.freq))
  }
  if (length(noise.ts) > 2) {
    noise.ts = noise.ts[2:(length(noise.ts) -1)]
  } else {
    noise.ts = c()
  }
  noise.x = sample(1:limit,length(noise.ts),replace=T)
  noise.y = sample(1:limit,length(noise.ts),replace=T)
  
  # 组合信号和噪声
  observations = data.frame(
    x = c(all.lines2d$x.observ, noise.x),
    y = c(all.lines2d$y.observ, noise.y),
    ts = c(all.lines2d$ts.observ, noise.ts),
    z = c(rep(1,length(all.lines2d$x.observ)), rep(0,length(noise.x))
  )
  observations = observations[order(observations$ts),]
  row.names(observations) = 1:nrow(observations)
  return(observations)
}

# 生成数据集（使用原始参数）
sim_data <- stomach.sim2d(
  limit = 8,       # 空间网格大小
  time.max = 80,   # 最大时间
  ts = 1,          # 起始时间
  x = 1, y = 1,    # 起始坐标
  u.x = 1, u.y = 1,# 传播方向
  p = 0.1,         # 信号丢失概率
  noise.freq = 1,  # 噪声频率
  miu = 1,         # 时间间隔均值
  sigma = 0.1,     # 时间间隔标准差
  gap = 30,        # 信号间隔
  ratio = 2        # 时间缩放因子
)

# 保存为CSV
write.csv(sim_data, "stomach_simulation_data.csv", row.names = FALSE)