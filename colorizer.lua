
require 'torch'
require 'image'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Transfers palette from one image to another.')
cmd:text()
cmd:option('-palette_image', 'gradients.png',
           'Image with desirable palette.')
cmd:option('-colorized_image', 'sausage.jpg',
           'Image to apply the palette.')
cmd:option('-output_image', 'yummy.png',
           'Where to save the result.')
cmd:option('-recolor_strength', 1,
           'Color transfer strength:\n\t\t\t0 = original colors,\n\t\t\t1 = apply the palette at full strength,\n\t\t\t-x...0...1...y = space to experiment')
cmd:option('-color_function', 'hsl-polar',
           'Color matching function: chol, pca, sym / mkl, rgb, xyz, lab, lms, hsl, hsl-full, hsl-polar, hsl-polar-full, idt, idt-mean, rgb-hist, lab-rgb, chol-pca, chol-sym, exp1')
cmd:option('-prefer_torch', false,
           'Prefer Torch calculations on single core (can be faster on some machines).')

cmd:text()

local params = cmd:parse(arg)
local prefer_Torch = params.prefer_torch or torch.getnumcores() > 1


local function main(params)
  print(params.palette_image, "=>", params.colorized_image)
  local content_img = image.load(params.palette_image, 3)
  local style_img = image.load(params.colorized_image, 3)

  local output_img = match_color(style_img, content_img, params.color_function)
  -- local output_img = content_img
  image.save(params.output_image, output_img)
--[[
  for a = -1, 3.11, 0.5
  do
    params.recolor_strength = a
    local output_img = match_color(style_img, content_img, params.color_function)
    local filename = params.output_image
    local ext = paths.extname(filename)
    local filename_mcvp = string.format('%s/%s_%+2.3f.%s', paths.dirname(filename), paths.basename(filename, ext),
                                        a, ext)
    image.save(filename_mcvp, output_img)
  end
--]]
end


local function haveNaNs(t)  -- Checks that tensor contains NaN values.
--[[
  for i = 1, t:size(1) do
    if math.abs(t[i]) > 0 then
    else
      return true
    end
  end
  return false
--]]
  return not torch.abs(t):ge(0):all()
end


-- Sorting v and x tensors by order of x, removing duplicates and NaNs.
local function clean_arrays_xv(x, v)
  -- Removing NaNs.
  local is_number = torch.abs(x):ge(0)
  local x, v = x[is_number], v[is_number]
  is_number = torch.abs(v):ge(0)
  x, v = x[is_number], v[is_number]

  -- Sorting
  local xnc
  x, xnc = torch.sort(x)
  v = v:index(1, xnc)

  -- Cleaning duplicates
  local xs1 = x:size(1)
  local px, n = math.sqrt(-1), torch.LongTensor(xs1)
--[[ -- Keep 1st duplicate x.
  local ix = 1
  for i = 1, xs1 do
    local xc = x[i]
    if xc ~= px then
      n[ix] = i
      ix = ix + 1
      px = xc
    end
  end
  n = n[{{1, ix - 1}}]
  x, v = x:index(1, n), v:index(1, n)
--]]
--[[ -- Keep last duplicate x.
  local ix = 0
  for i = 1, xs1 do
    local xc = x[i]
    if xc ~= px then
      ix = ix + 1
      px = xc
    end
    n[ix] = i
  end
  n = n[{{1, ix}}]
  x, v = x:index(1, n), v:index(1, n)
--]]
-- --[[ -- Average Vs, corresponding to same Xs.
  local ix, ax, va = 0, 0, torch.Tensor(xs1)
  for i = 1, xs1 do
    local xc = x[i]
    if xc ~= px then
      ix = ix + 1
      px = xc
      ax = 1
      n[ix] = i
      va[ix] = v[i]
    else
      ax = ax + 1
      va[ix] = (va[ix] * (ax - 1) + v[i]) / ax
    end
  end
  x, v = x:index(1, n[{{1, ix}}]), va[{{1, ix}}]
--]]

  return x, v
end


-- NaNs are zeroed in resulting tensor.
local function lin_interp(x, v, xq)
-- http://www.mathworks.com/help/matlab/ref/interp1.html
  local xqs1 = xq:size(1)
  if x:size(1) == 1 then  -- only 1 point, nothing to extrapolate
    if math.abs(v[1]) >= 0 then
      return torch.Tensor(xqs1):fill(v[1]) -- point-weighted, assuming v[n] = v[1]
--    return torch.mul(xq, v[1]):div(x[1]) -- zero-weighted, vq = [xq-0]/[x-0]*[v-0]
    else
      return torch.Tensor(xqs1):fill(0)
    end
  else
    local x, v = clean_arrays_xv(x, v)
    local xs1 = x:size(1)

    local s, a, vq = torch.Tensor(xs1), torch.Tensor(xs1), torch.zeros(xqs1) -- to not generate NaNs accidentally.
    local x_min, x_max = x[1], x[-1]
    local x_bin = (x_max - x_min) / (xs1 - 1)
    s[{{1, -2}}] = (v[{{2, -1}}] - v[{{1, -2}}]):cdiv(x[{{2, -1}}] - x[{{1, -2}}])
    a[{{1, -2}}] = torch.addcmul(v[{{1, -2}}], -x[{{1, -2}}], s[{{1, -2}}])
    local s1, sL = s[1], s[-2]
    local a1, aL = a[1], v[-1] - x_max * sL
    s[-1], a[-1] = sL, aL  -- [-1] should not be used (x>=x_max handled separately), but rounding errors can result in [-1]

    -- Finding maximum x values, less than equally spaced x[]
    local x_search, fMin = torch.Tensor(xs1), 1
    for i = 2, xs1 do
      local fMax = math.min(math.floor((x[i] - x_min) / x_bin) + 1, xs1)
      if fMax ~= fMin then
        x_search[{{fMin, fMax - 1}}]:fill(i - 1)
        fMin = fMax
      end
    end
    x_search[{{fMin, -1}}]:fill(xs1)

    if prefer_Torch == true then -- Torch parallel calculations should be faster in multicore environment.
      local xi = (xq - x_min):div(x_bin):floor():long():add(1)  -- Equally-spaced index
      local xq_numeric = torch.abs(xq):ge(0)  -- Numeric values flag
      for xqi = 1, xqs1 do
        local c = xq[xqi]
        if c <= x_min then               -- extrapolate below
          vq[xqi] = c * s1 + a1
        elseif c >= x_max then           -- extrapolate above
          vq[xqi] = c * sL + aL
        elseif xq_numeric[xqi] == 1 then -- interpolate, NaNs are ignored
          for i = x_search[xi[xqi]], 1, -1 do
            if c >= x[i] then
              vq[xqi] = c * s[i] + a[i]
              break
            end
          end
        end
      end
    else -- Lua variant should be faster on single core, because calculations are partially skipped, depending on images.
      for xqi = 1, xqs1 do
        local c = xq[xqi]
        if c <= x_min then           -- extrapolate below
          vq[xqi] = c * s1 + a1
        elseif c >= x_max then       -- extrapolate above
          vq[xqi] = c * sL + aL
        elseif math.abs(c) >= 0 then -- interpolate, NaNs are ignored
          for i = x_search[ math.floor((c - x_min) / x_bin) + 1 ], 1, -1 do
            if c >= x[i] then
              vq[xqi] = c * s[i] + a[i]
              break
            end
          end
        end
      end
    end   -- Single/multicore
    return vq
  end
end


-- Faster variant, mean-weighted interpolation.
local function lin_interp_mean(x, v, xq)
-- http://www.mathworks.com/help/matlab/ref/interp1.html
  local xqs1 = xq:size(1)
  if x:size(1) == 1 then
    if math.abs(v[1]) >= 0 then
      return torch.Tensor(xqs1):fill(v[1])
    else
      return torch.Tensor(xqs1):fill(0)
    end
  else
    -- Removing NaNs.
    local is_number = torch.abs(x):ge(0)
    local x, v = x[is_number], v[is_number]
    is_number = torch.abs(v):ge(0)
    x, v = x[is_number], v[is_number]
    local xs1 = x:size(1)

    local s, a, vq = torch.Tensor(xs1), torch.Tensor(xs1), torch.zeros(xqs1)
    local x_min, x_max = x:min(), x:max()
    local x_bin = (x_max - x_min) / (xs1 - 1)

    s[{{1, -2}}] = (v[{{2, -1}}] - v[{{1, -2}}]):div(x_bin)
    a[{{1, -2}}] = torch.addcmul(v[{{1, -2}}], -x[{{1, -2}}], s[{{1, -2}}])

    local s1, sL = s[1], s[-2]
    local a1, aL = a[1], v[-1] - x_max * sL
    s[-1], a[-1] = sL, aL

    if prefer_Torch == true then -- Torch parallel calculations should be faster in multicore environment.
      local xi = (xq - x_min):div(x_bin):floor():long():add(1)
      local xq_numeric = torch.abs(xq):ge(0)  -- Numeric values flag
      for xqi = 1, xqs1 do
        local c = xq[xqi]
        if c <= x_min then               -- extrapolate below
          vq[xqi] = c * s1 + a1
        elseif c >= x_max then           -- extrapolate above
          vq[xqi] = c * sL + aL
        elseif xq_numeric[xqi] == 1 then -- interpolate, NaNs are ignored
          local i = xi[xqi]
          vq[xqi] = c * s[i] + a[i]
        end
      end
    else -- Lua variant should be faster on single core, because calculations are partially skipped, depending on images.
      for xqi = 1, xqs1 do
        local c = xq[xqi]
        if c <= x_min then           -- extrapolate below
          vq[xqi] = c * s1 + a1
        elseif c >= x_max then       -- extrapolate above
          vq[xqi] = c * sL + aL
        elseif math.abs(c) >= 0 then -- interpolate, NaNs are ignored
          local i = math.floor((c - x_min) / x_bin) + 1
          vq[xqi] = c * s[i] + a[i]
        end
      end
    end   -- Single/multicore
    return vq
  end
end


-- Faster variant for tensors without NaNs and sorted and equally spaced X.
local function lin_interp_r(x, v, xq)
-- http://www.mathworks.com/help/matlab/ref/interp1.html
  local xs1, xqs1 = x:size(1), xq:size(1)
  if xs1 == 1 then
    return torch.Tensor(xqs1):fill(v[1])
  else
    local s, a, vq = torch.Tensor(xs1), torch.Tensor(xs1), torch.Tensor(xqs1)
    local x_min, x_max = x[1], x[-1]
    local x_bin = (x_max - x_min) / (xs1 - 1)
    s[{{1, -2}}] = (v[{{2, -1}}] - v[{{1, -2}}]):div(x_bin)
    a[{{1, -2}}] = torch.addcmul(v[{{1, -2}}], -x[{{1, -2}}], s[{{1, -2}}])

    local s1, sL = s[1], s[-2]
    local a1, aL = a[1], v[-1] - x_max * sL
    s[-1], a[-1] = sL, aL

    if prefer_Torch == true then -- Torch parallel calculations should be faster in multicore environment.
      local xi = (xq - x_min):div(x_bin):floor():long():add(1)
      for xqi = 1, xqs1 do
        local c = xq[xqi]
        if c <= x_min then     -- extrapolate below
          vq[xqi] = c * s1 + a1
        elseif c >= x_max then -- extrapolate above
          vq[xqi] = c * sL + aL
        else                   -- interpolate
          local i = xi[xqi]
          vq[xqi] = c * s[i] + a[i]
        end
      end
    else -- Lua variant should be faster on single core, because calculations are partially skipped, depending on images.
      for xqi = 1, xqs1 do
        local c = xq[xqi]
        if c <= x_min then     -- extrapolate below
          vq[xqi] = c * s1 + a1
        elseif c >= x_max then -- extrapolate above
          vq[xqi] = c * sL + aL
        else                   -- interpolate
          local i = math.floor((c - x_min) / x_bin) + 1
          vq[xqi] = c * s[i] + a[i]
        end
      end
    end   -- Single/multicore
    return vq
  end
end


-- Direct reimplementation in Torch of https://github.com/scipy/scipy/blob/master/scipy/linalg/decomp_svd.py#L214
local function ml_orth(A)
-- http://www.mathworks.com/help/matlab/ref/orth.html
-- Construct an orthonormal basis for the range of A using SVD
  local u, s, v = torch.svd(A, 'S')
  local M, N = A:size(1), A:size(2)
  local eps = 1e-10
  local tol = math.max(M, N) * s:max() * eps
  local num = torch.gt(s, tol):sum()
  return u[{{}, {1, num}}]
end


-- Direct reimplementation in Torch of https://github.com/frcs/colour-transfer, (c) F. Pitie 2007.
local function pdf_transfer1D(pX,pY)
  local nbins =  pX:size(1)
  local eps = 1e-6 -- small damping term that faciliates the inversion

  local PX = torch.cumsum(pX + eps)
  PX = torch.div(PX, PX[-1])

  local PY = torch.cumsum(pY + eps)
  PY = torch.div(PY, PY[-1])

  -- inversion
  local f = lin_interp(PY, torch.range(0, nbins-1), PX)
  f[torch.le(PX, PY[1])] = 0
  f[torch.ge(PX, PY[-1])] = nbins-1

  -- Currently, lin_interp zeroes NaNs, therefore this message is useless.
  --if haveNaNs(f) then print("pdf_transfer1D: NaN values have been generated.") end

  return f
end


-- Direct reimplementation in Torch of https://github.com/frcs/colour-transfer, (c) F. Pitie 2007.
local function pdf_transfer(D0, D1, Rotations, varargin)
  local relaxation = varargin or 1.0  -- colorization level
  local nb_iterations = Rotations:size(1)
  local eps = 1e-10

  local hist_points = 300   -- histogram precision, calculation time is proportional

  for it = 1, nb_iterations do
    print(string.format('IDT iteration %02d / %02d', it, nb_iterations))

    local R = Rotations[it]
    local nb_projs = R:size(1) -- 6

    -- apply rotation
    local D0R = R * D0
    local D1R = R * D1
    local D0R_ = torch.Tensor(D0R:size()):zero()

    -- get the marginals, match them, and apply transformation
    for i = 1, nb_projs do
      print(string.format('Projection %d / %d', i, nb_projs))

      -- get the data range
      local datamin = math.min(D0R[i]:min(), D1R[i]:min()) - eps
      local datamax = math.max(D0R[i]:max(), D1R[i]:max()) + eps
      local u = torch.linspace(datamin, datamax, hist_points)

      -- get the projections
      local p0R = torch.histc(D0R[i], hist_points, datamin, datamax)
      local p1R = torch.histc(D1R[i], hist_points, datamin, datamax)

      -- get the transport map
      local f = pdf_transfer1D(p0R, p1R)

      -- apply the mapping
      D0R_[i] = (lin_interp(u, f, D0R[i])-1) / (hist_points - 1) * (datamax-datamin) + datamin
    end

    D0:add(torch.inverse(R):t() * (D0R_ - D0R) * relaxation)  -- D0 = relaxation * (R \ (D0R_ - D0R)) + D0;
  end

  return D0
end


-- Direct reimplementation in Torch of https://github.com/frcs/colour-transfer, (c) F. Pitie 2007.
local function pdf_transfer1D_mean_weighted(pX,pY)
-- With mean-weigthed linear interpolation
  local nbins = pX:size(1)
  local eps = 1e-6 -- small damping term that faciliates the inversion

  local PX = torch.cumsum(pX + eps)
  PX = torch.div(PX, PX[-1])

  local PY = torch.cumsum(pY + eps)
  PY = torch.div(PY, PY[-1])

  -- inversion
  local f = lin_interp_mean(PY, torch.range(0, nbins - 1), PX)
  f[torch.le(PX, PY[1])] = 0
  f[torch.ge(PX, PY[-1])] = nbins - 1

  -- Currently, lin_interp zeroes NaNs, therefore this message is useless.
  --if haveNaNs(f) then print("pdf_transfer1D: NaN values have been generated.") end

  return f
end


-- Direct reimplementation in Torch of https://github.com/frcs/colour-transfer, (c) F. Pitie 2007.
local function pdf_transfer_mean_weighted(D0, D1, R, nb_iterations, varargin)
-- With mean-weigthed linear interpolation
  local relaxation = varargin or 1.0  -- colorization level
  local eps = 1e-10

  local hist_points = 300   -- histogram precision, calculation time is proportional

  for it = 1, nb_iterations do
    print(string.format('IDT iteration %02d / %02d', it, nb_iterations))

    local nb_projs = R:size(1) -- 6

    -- apply rotation
    local D0R = R * D0
    local D1R = R * D1
    local D0R_ = torch.Tensor(D0R:size()):zero()

    -- get the marginals, match them, and apply transformation
    for i = 1, nb_projs do
      print(string.format('Projection %d / %d', i, nb_projs))

      -- get the data range
      local datamin = math.min(D0R[i]:min(), D1R[i]:min()) - eps
      local datamax = math.max(D0R[i]:max(), D1R[i]:max()) + eps
      local u = torch.linspace(datamin, datamax, hist_points)

      -- get the projections
      local p0R = torch.histc(D0R[i], hist_points, datamin, datamax)
      local p1R = torch.histc(D1R[i], hist_points, datamin, datamax)

      -- get the transport map
      local f = pdf_transfer1D_mean_weighted(p0R, p1R)

      -- apply the mapping
      D0R_[i] = (lin_interp_mean(u, f, D0R[i]) - 1) / (hist_points - 1) * (datamax - datamin) + datamin
    end

    D0:add(torch.inverse(R):t() * (D0R_ - D0R) * relaxation)  -- D0 = relaxation * (R \ (D0R_ - D0R)) + D0;
  end

  return D0
end


local function reshape_histogram(channel_s, channel_d, hist_points)
-- Scales destination histogram by shape of source histogram.
-- Inspired by https://github.com/frcs/colour-transfer
  local eps = 1e-10

  -- Making histograms
  local range_min_s, range_max_s = channel_s:min(), channel_s:max()   -- source range
  local hist_points_s = torch.linspace(range_min_s, range_max_s, hist_points)
  local hist_s = torch.histc(channel_s, hist_points, range_min_s, range_max_s)   -- number of values within points' ranges

  local range_min_d, range_max_d = channel_d:min(), channel_d:max()
  local hist_points_d = torch.linspace(range_min_d, range_max_d, hist_points)
  local hist_d = torch.histc(channel_d, hist_points, range_min_d, range_max_d)

  -- Normalizing histograms
  local hist_s_n = (hist_s + eps):div(hist_s:max() + eps)
  local hist_d_n = (hist_d + eps):div(hist_d:max() + eps)

  -- Reshaping histogram
  local hist_r = torch.cdiv(hist_d_n + eps, hist_s_n + eps)

  -- Normalizing scaling factor, more relaxed for smaller histograms
  hist_r:log()
  local hist_r_n = torch.abs(hist_r):max() / hist_points + eps
  hist_r:add(eps):div(hist_r_n)

  -- Weighting scaling coefficients with new histogram
  local shape_r = lin_interp_r(hist_points_d, hist_r, channel_d)

  -- Scaling image channel
  local mean_c_s, mean_c_d = channel_s:mean(), channel_d:mean()
  local std_c_s, std_c_d = channel_s:std(), channel_d:std()
  local scale_r = torch.Tensor(channel_d:size()):fill(std_c_s / std_c_d):cpow(shape_r)
  local channel_r = (channel_d - mean_c_d):cmul(scale_r):add(mean_c_s)

  return channel_r
end


function match_color(target_img, source_img, mode, eps)
  -- Matches the colour distribution of the target image to that of the source image
  -- using a linear transform.
  -- Images are expected to be of form (c,h,w) and float in [0,1].
  -- Modes are chol, pca, sym / mkl, rgb, xyz, lab, lms, hsl, hsl-polar, labrgb, cholMpca, cholMsym, exp1.

--  if target_img:equal(source_img) then return target_img end
  mode = mode or 'hsl-polar'
  eps = eps or 1e-5

  if mode == 'lab' then
    -- Color transfer between images
    -- https://github.com/jrosebr1/color_transfer
    -- https://www.researchgate.net/publication/220518215_Color_Transfer_between_Images
    -- https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf
    local s_lab = image.rgb2lab(source_img):view(source_img:size(1), source_img[1]:nElement())
    local t_lab = image.rgb2lab(target_img):view(target_img:size(1), target_img[1]:nElement())
    -- Is range -100...100?
    -- print(s_lab:min(), s_lab:max())
    -- print(t_lab:min(), t_lab:max())
    local sMean, sStd = s_lab:mean(2), s_lab:std(2, true)
    local tMean, tStd = t_lab:mean(2), t_lab:std(2, true)
    local tCol = (t_lab - tMean:expandAs(t_lab)):cmul(sStd:cdiv(tStd):expandAs(t_lab)) + sMean:expandAs(t_lab)
    return image.lab2rgb(tCol:viewAs(target_img):clamp(0, 255)):clamp(0, 1)
  elseif mode == 'rgb' then
    local sMean, sStd = source_img:mean(3):mean(2), source_img:view(source_img:size(1), source_img[1]:nElement()):std(2, true):view(3, 1, 1)
    local tMean, tStd = target_img:mean(3):mean(2), target_img:view(target_img:size(1), target_img[1]:nElement()):std(2, true):view(3, 1, 1)
    local tCol = (target_img - tMean:expandAs(target_img)):cmul(sStd:cdiv(tStd):expandAs(target_img)) + sMean:expandAs(target_img)
    return tCol:clamp(0, 1)
  elseif mode == 'xyz' then
    -- Coefficients from https://github.com/THEjoezack/ColorMine/blob/master/ColorMine/ColorSpaces/Conversions/XyzConverter.cs
    -- local xyz_s = torch.Tensor(source_img:size(1),source_img:size(2),source_img:size(3))
    -- xyz_s[1] = torch.mul(source_img[1], 0.4124) + torch.mul(source_img[2], 0.3576) + torch.mul(source_img[3], 0.1805)
    -- xyz_s[2] = torch.mul(source_img[1], 0.2126) + torch.mul(source_img[2], 0.7152) + torch.mul(source_img[3], 0.0722)
    -- xyz_s[3] = torch.mul(source_img[1], 0.0193) + torch.mul(source_img[2], 0.1192) + torch.mul(source_img[3], 0.9505)

    -- X = r * 0.4124 + g * 0.3576 + b * 0.1805;   R = x *  3.2406 + y * -1.5372 + z * -0.4986;
    -- Y = r * 0.2126 + g * 0.7152 + b * 0.0722;   G = x * -0.9689 + y *  1.8758 + z *  0.0415;
    -- Z = r * 0.0193 + g * 0.1192 + b * 0.9505;   B = x *  0.0557 + y * -0.2040 + z *  1.0570;
    local rgb_xyz_mat = torch.Tensor({{0.4124, 0.3576, 0.1805},
                                      {0.2126, 0.7152, 0.0722},
                                      {0.0193, 0.1192, 0.9505}})
    local xyz_s = (rgb_xyz_mat * source_img:view(source_img:size(1), source_img[1]:nElement()))
    local xyz_t = (rgb_xyz_mat * target_img:view(target_img:size(1), target_img[1]:nElement()))

    local sMean, sStd = xyz_s:mean(2), xyz_s:std(2, true)
    local tMean, tStd = xyz_t:mean(2), xyz_t:std(2, true)
    local tCol = (xyz_t - tMean:expandAs(xyz_t)):cmul(sStd:cdiv(tStd):expandAs(xyz_t)) + sMean:expandAs(xyz_t)

    local xyz_rgb_mat = torch.Tensor({{ 3.2406, -1.5372, -0.4986},
                                      {-0.9689,  1.8758,  0.0415},
                                      { 0.0557, -0.2040,  1.0570}})
    tCol = (xyz_rgb_mat * tCol):viewAs(target_img)
    return tCol:clamp(0, 1)
  elseif mode == 'lms' then
    -- https://www.researchgate.net/publication/220518215_Color_Transfer_between_Images
    -- https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf
    --   r       g       b
    -- l 0.3811  0.5783  0.0402
    -- m 0.1967  0.7244  0.0782
    -- s 0.0241  0.1288  0.8444
    -- l,m,s = log(l,m,s)  -- Decimal logarithm is used in original paper, but
                           -- it seems that the function can be done with natural logarithms, and
                           -- without division/multiplication by log(10) it should be a little faster.
    --   l   m   s          l         m         s
    -- l 1,  1,  1        l 1/sqr(3), 0,        0
    -- m 1,  1, -2   ,    a 0,        1/sqr(6), 0
    -- s 1, -1,  0        b 0,        0,        1/sqr(2)

    -- Lab = (t - Mean(t)) * Std(s) / Std(t) + Mean(s)

    --   l         a         b                 l   m   s
    -- l sqr(3)/3, 0,        0               l 1,  1,  1
    -- m 0,        sqr(6)/6, 0          ,    m 1,  1, -1
    -- s 0,        0,        sqr(2)/2        s 1, -2,  0
    -- l,m,s = 10^{l,m,s}   -- e^{l,m,s} ?
    --    l       m       s
    -- r  4.4679 -3.5873  0.1193
    -- g -1.2186  2.3809 -0.1624
    -- b  0.0497 -0.2439  1.2045

    local rgb_lms_mat = torch.Tensor({{0.3811, 0.5783, 0.0402},
                                      {0.1967, 0.7244, 0.0782},
                                      {0.0241, 0.1288, 0.8444}})
    local lms_mat2 = torch.Tensor({{1.0,  1.0,  1.0},
                                   {1.0,  1.0, -2.0},
                                   {1.0, -1.0,  0.0}})
    local lms_mat3 = torch.Tensor({{1/math.sqrt(3), 0.0,            0.0},
                                   {0.0,            1/math.sqrt(6), 0.0},
                                   {0.0,            0.0,            1/math.sqrt(2)}})
    local lms_s = (rgb_lms_mat * source_img:view(source_img:size(1), source_img[1]:nElement())):add(eps):log() -- / math.log(10)
    lms_s = lms_mat3 * (lms_mat2 * lms_s)
    local lms_t = (rgb_lms_mat * target_img:view(target_img:size(1), target_img[1]:nElement())):add(eps):log() -- / math.log(10)
    lms_t = lms_mat3 * (lms_mat2 * lms_t)

    local sMean, sStd = lms_s:mean(2), lms_s:std(2, true)
    local tMean, tStd = lms_t:mean(2), lms_t:std(2, true)
    local tCol = (lms_t - tMean:expandAs(lms_t)):cmul(sStd:cdiv(tStd):expandAs(lms_t)) + sMean:expandAs(lms_t)

    local lms_mat4 = torch.Tensor({{math.sqrt(3)/3, 0.0,            0.0},
                                   {0.0,            math.sqrt(6)/6, 0.0},
                                   {0.0,            0.0,            math.sqrt(2)/2}})
    local lms_mat5 = torch.Tensor({{1.0,  1.0,  1.0},
                                   {1.0,  1.0, -1.0},
                                   {1.0, -2.0,  0.0}})
    local lms_rgb_mat = torch.Tensor({{ 4.4679, -3.5873,  0.1193},
                                      {-1.2186,  2.3809, -0.1624},
                                      { 0.0497, -0.2439,  1.2045}})
    tCol = (lms_mat5 * (lms_mat4 * tCol)):exp()             -- decimal: tCol:mul(math.log(10)):exp() --??? - 1e-5
    local lms_rgb = (lms_rgb_mat * tCol):viewAs(target_img)
    return lms_rgb:clamp(0, 1)
  elseif mode == 'hsl-full' then
    -- Hue scaling in Cartesian coordinates, saturation-independent
    local s_hsl = image.rgb2hsl(source_img):view(source_img:size(1), source_img[1]:nElement())  -- 0...1 range?
    local t_hsl = image.rgb2hsl(target_img):view(target_img:size(1), target_img[1]:nElement())

    s_hsl[1]:mul(math.pi * 2):remainder(math.pi * 2)  -- a % 2π reduces sine error with angles outside 0...2π range
    t_hsl[1]:mul(math.pi * 2):remainder(math.pi * 2)
    local s_cos = torch.cos(s_hsl[1])
    local t_cos = torch.cos(t_hsl[1])
    s_hsl[1]:sin()
    t_hsl[1]:sin()
--[[
    local da = 1e0
    for a = -3601, 7201, da do
      b = a / 360 * (math.pi * 2)
      s = math.sin(b % (math.pi * 2))
      c = math.cos(b % (math.pi * 2))

      r = math.asin(s)
      r1 = r / (2 * math.pi) * 360
      if c < 0 then r = math.pi - r end
      r = r % (2 * math.pi)
      r = r / (2 * math.pi) * 360

      rc = math.acos(c)
      rc1 = rc / (2 * math.pi) * 360
      if s < 0 then rc = -rc end
      rc = rc % (2 * math.pi)
      rc = rc / (2 * math.pi) * 360

--      if a % 45 < da then print(a) end
      if (math.abs(a % 360 - r) > 0.1) or (math.abs(a % 360 - rc) > 0.1) then
        print(a, s, c, r, rc, a % 360 - r, a % 360 - rc, a % 360 - r % 360)
      end
    end
    os.exit()
--]]

    -- Independent hue scaling
    local scMean, scStd = s_cos:mean(), s_cos:var(1, true)[1]
    local tcMean, tcStd = t_cos:mean(), t_cos:var(1, true)[1]
    local sMean, sStd = s_hsl:mean(2), torch.Tensor(3, 1)
    local tMean, tStd = t_hsl:mean(2), torch.Tensor(3, 1)
    sStd[1], sStd[2], sStd[3] = torch.var(s_hsl[1], 1, true), torch.std(s_hsl[2], 1, true), torch.std(s_hsl[3], 1, true)
    tStd[1], tStd[2], tStd[3] = torch.var(t_hsl[1], 1, true), torch.std(t_hsl[2], 1, true), torch.std(t_hsl[3], 1, true)
    local tCol = torch.Tensor(3, t_hsl:size(2))
    tCol[1] = (t_hsl[1] - tMean[1][1]):mul((sStd[1][1] / tStd[1][1]) ^ 1.0):add(sMean[1][1]) -- 3 ≈ colorize, 1 = variance, 0.5 = std, 0 = relaxed colorization
    tCol[2] = (t_hsl[2] - tMean[2][1]):mul(sStd[2][1] / tStd[2][1]):add(sMean[2][1])         --               variance feels most balanced to me
    tCol[3] = (t_hsl[3] - tMean[3][1]):mul(sStd[3][1] / tStd[3][1]):add(sMean[3][1])
    local tcRes = (t_cos - tcMean):mul((scStd / tcStd)               ^ 1.0):add(scMean)

    -- Normalizing hue vector
    local tHueScale = torch.pow(tCol[1], 2):add(torch.pow(tcRes, 2)):sqrt()
    tCol[1]:cdiv(tHueScale)
    tcRes:cdiv(tHueScale)

    -- Restoring hue angle
    tCol[1]:clamp(-1, 1) -- or asin / acos may produce "not a number" overflows
    tcRes:clamp(-1, 1)                    -- angle  -90°...0°...90°...180°  181°...269° 270°
    local sn = torch.lt(tCol[1], 0)       -- sine    -1 ...0 ... 1 ...  0   ~-0 ...~-1   -1
    local cn = torch.lt(tcRes, 0)         -- cosine   0 ...1 ... 0 ... -1   ~-1 ...~-0    0
    tCol[1]:asin()                        -- asin   -90°...0°...90°...  0°   -1 ...-89  -90°
    tcRes:acos()                          -- acos    90°...0°...90°...180°  179°... 91   90°
    tCol[1][cn] = math.pi - tCol[1][cn]   --        -90°...0°...90°...180°  181°...269  -90°
    tcRes[sn] = -tcRes[sn]                --        -90°...0°...90°...180° -179°...-91  -90°
    tCol[1]:remainder(math.pi * 2)        -- a % 2π 270°...0°...90°...180°  181°...269  270°
    tcRes:remainder(math.pi * 2)          -- always 360 => 0, safe to use sqrt(a*b)

    -- Merging angles, restored from both sine and cosine, to improve precision
    -- 1) Simple variant, fastest, but makes even more errors (compared to "original > original") than log-mean
    --tCol[1]:cmul(tcRes):sqrt()
    -- --
    -- 2) Mean / logarithmic mean variant
    -- Rotating by π to remove possible rounding errors at 0-360 point
    local m180 = (math.pi - tCol[1]):abs():ge(math.pi / 2)   -- mask to replace with rotated means
    local tCol180 = torch.add(tCol[1], math.pi):remainder(math.pi * 2)
    local tRes180 = torch.add(tcRes, math.pi):remainder(math.pi * 2)
    -- 2.1) Mean, seems to make less errors
    tCol180:add(tRes180):div(2)
    tCol[1]:add(tcRes):div(2)
    -- 2.2) Logarithmic mean, seems to make more errors, therefore probably doesn't make sense at all
    --tCol180:cmul(tRes180):sqrt()
    --tCol[1]:cmul(tcRes):sqrt()
    -- --
    tCol180:add(math.pi):remainder(math.pi * 2)  -- Rotating back
    tCol[1][m180] = tCol180[m180]                -- and combining error-free halves

    tCol[1]:div(math.pi * 2)
    return image.hsl2rgb(tCol:clamp(0, 1):viewAs(target_img)):clamp(0, 1)
  elseif mode == 'hsl' then
    -- Hue scaling in Cartesian coordinates, saturation-weighted
    local s_hsl = image.rgb2hsl(source_img):view(source_img:size(1), source_img[1]:nElement())  -- 0...1 range?
    local t_hsl = image.rgb2hsl(target_img):view(target_img:size(1), target_img[1]:nElement())

    s_hsl[1]:mul(math.pi * 2):remainder(math.pi * 2)  -- a % 2π reduces sine error with angles outside 0...2π range
    t_hsl[1]:mul(math.pi * 2):remainder(math.pi * 2)
    local s_cos = torch.cos(s_hsl[1]):cmul(s_hsl[2])
    local t_cos = torch.cos(t_hsl[1]):cmul(t_hsl[2])
    s_hsl[1]:sin():cmul(s_hsl[2])
    t_hsl[1]:sin():cmul(t_hsl[2])

    -- Independent hue scaling
    local scMean, scStd = s_cos:mean(), s_cos:var(1, true)[1]
    local tcMean, tcStd = t_cos:mean(), t_cos:var(1, true)[1]
    local sMean, sStd = s_hsl:mean(2), torch.Tensor(3, 1)
    local tMean, tStd = t_hsl:mean(2), torch.Tensor(3, 1)
    sStd[1], sStd[2], sStd[3] = torch.var(s_hsl[1], 1, true), torch.std(s_hsl[2], 1, true), torch.std(s_hsl[3], 1, true)
    tStd[1], tStd[2], tStd[3] = torch.var(t_hsl[1], 1, true), torch.std(t_hsl[2], 1, true), torch.std(t_hsl[3], 1, true)
    local tCol = torch.Tensor(3, t_hsl:size(2))
    tCol[1] = (t_hsl[1] - tMean[1][1]):mul((sStd[1][1] / tStd[1][1]) ^ 1.0):add(sMean[1][1]) -- 3 ≈ colorize, 1 = variance, 0.5 = std, 0 = relaxed colorization
    tCol[2] = (t_hsl[2] - tMean[2][1]):mul(sStd[2][1] / tStd[2][1]):add(sMean[2][1])         --               variance feels most balanced to me
    tCol[3] = (t_hsl[3] - tMean[3][1]):mul(sStd[3][1] / tStd[3][1]):add(sMean[3][1])
    local tcRes = (t_cos - tcMean):mul((scStd / tcStd)               ^ 1.0):add(scMean)

    -- Normalizing hue vector
    local tHueScale = torch.pow(tCol[1], 2):add(torch.pow(tcRes, 2)):sqrt()
    tCol[1]:cdiv(tHueScale)
    tcRes:cdiv(tHueScale)

    -- Restoring hue angle
    tCol[1]:clamp(-1, 1) -- or asin / acos may produce "not a number" overflows
    tcRes:clamp(-1, 1)                    -- angle  -90°...0°...90°...180°  181°...269° 270°
    local sn = torch.lt(tCol[1], 0)       -- sine    -1 ...0 ... 1 ...  0   ~-0 ...~-1   -1
    local cn = torch.lt(tcRes, 0)         -- cosine   0 ...1 ... 0 ... -1   ~-1 ...~-0    0
    tCol[1]:asin()                        -- asin   -90°...0°...90°...  0°   -1 ...-89  -90°
    tcRes:acos()                          -- acos    90°...0°...90°...180°  179°... 91   90°
    tCol[1][cn] = math.pi - tCol[1][cn]   --        -90°...0°...90°...180°  181°...269  -90°
    tcRes[sn] = -tcRes[sn]                --        -90°...0°...90°...180° -179°...-91  -90°
    tCol[1]:remainder(math.pi * 2)        -- a % 2π 270°...0°...90°...180°  181°...269  270°
    tcRes:remainder(math.pi * 2)          -- always 360 => 0, safe to use sqrt(a*b)

    -- Merging angles, restored from both sine and cosine, to improve precision
    -- 1) Simple variant, fastest, but makes even more errors (compared to "original > original") than log-mean
    --tCol[1]:cmul(tcRes):sqrt()
    -- --
    -- 2) Mean / logarithmic mean variant
    -- Rotating by π to remove possible rounding errors at 0-360 point
    local m180 = (math.pi - tCol[1]):abs():ge(math.pi / 2)   -- mask to replace with rotated means
    local tCol180 = torch.add(tCol[1], math.pi):remainder(math.pi * 2)
    local tRes180 = torch.add(tcRes, math.pi):remainder(math.pi * 2)
    -- 2.1) Mean, seems to make less errors
    tCol180:add(tRes180):div(2)
    tCol[1]:add(tcRes):div(2)
    -- 2.2) Logarithmic mean, seems to make more errors, therefore probably doesn't make sense at all
    --tCol180:cmul(tRes180):sqrt()
    --tCol[1]:cmul(tcRes):sqrt()
    -- --
    tCol180:add(math.pi):remainder(math.pi * 2)  -- Rotating back
    tCol[1][m180] = tCol180[m180]                -- and combining error-free halves

    tCol[1]:div(math.pi * 2)
    return image.hsl2rgb(tCol:clamp(0, 1):viewAs(target_img)):clamp(0, 1)
  elseif mode == 'hsl-tint' then
    -- Scaling in Cartesian coordinates, saturation is scaled together with hue
    local s_hsl = image.rgb2hsl(source_img):view(source_img:size(1), source_img[1]:nElement())
    local t_hsl = image.rgb2hsl(target_img):view(target_img:size(1), target_img[1]:nElement())

    -- Hue-saturation vector projections
    s_hsl[1]:mul(math.pi * 2)
    t_hsl[1]:mul(math.pi * 2)
    local s_cos = torch.cos(s_hsl[1]):cmul(s_hsl[2])
    local t_cos = torch.cos(t_hsl[1]):cmul(t_hsl[2])
    s_hsl[1]:sin():cmul(s_hsl[2])
    t_hsl[1]:sin():cmul(t_hsl[2])
    s_hsl[2] = s_cos
    t_hsl[2] = t_cos

    -- Scaling
    local sMean, sStd = s_hsl:mean(2):squeeze(), torch.Tensor(3, 1)
    local tMean, tStd = t_hsl:mean(2):squeeze(), torch.Tensor(3, 1)
    sStd[1], sStd[2], sStd[3] = torch.var(s_hsl[1], 1, true), torch.var(s_hsl[2], 1, true), torch.std(s_hsl[3], 1, true)
    tStd[1], tStd[2], tStd[3] = torch.var(t_hsl[1], 1, true), torch.var(t_hsl[2], 1, true), torch.std(t_hsl[3], 1, true)
    local tCol = torch.Tensor(3, t_hsl:size(2))
    tCol[1] = (t_hsl[1] - tMean[1]):mul((sStd[1][1] / tStd[1][1]) ^ 0.5):add(sMean[1])
    tCol[2] = (t_hsl[2] - tMean[2]):mul((sStd[2][1] / tStd[2][1]) ^ 0.5):add(sMean[2])
    tCol[3] = (t_hsl[3] - tMean[3]):mul( sStd[3][1] / tStd[3][1]):add(sMean[3])

    -- Splitting hue and saturation back
    local tSat = torch.pow(tCol[1], 2):add(torch.pow(tCol[2], 2)):sqrt()
    tCol[1]:cdiv(tSat)
    tCol[2]:cdiv(tSat)

    -- Restoring hue angle
    tCol[1]:clamp(-1, 1)
    tCol[2]:clamp(-1, 1)
    local sn = torch.lt(tCol[1], 0)
    local cn = torch.lt(tCol[2], 0)
    tCol[1]:asin()
    tCol[2]:acos()
    tCol[1][cn] = math.pi - tCol[1][cn]
    tCol[2][sn] = -tCol[2][sn]
    tCol[1]:remainder(math.pi * 2)
    tCol[2]:remainder(math.pi * 2)

    -- Averaging angles, restored from sine and cosine, to improve precision
    local m180 = (math.pi - tCol[1]):abs():ge(math.pi / 2)             -- Rotated half mask
    local tCol180 = torch.add(tCol[1], math.pi):remainder(math.pi * 2) -- Rotating by π to remove possible rounding errors at 0-360 point
    local tRes180 = torch.add(tCol[2], math.pi):remainder(math.pi * 2)
    tCol180:add(tRes180):div(2)
    tCol[1]:add(tCol[2]):div(2)
    tCol180:add(math.pi):remainder(math.pi * 2)  -- Rotating back
    tCol[1][m180] = tCol180[m180]                -- and combining error-free halves

    tCol[1]:div(math.pi * 2)
    tCol[2] = tSat
    return image.hsl2rgb(tCol:clamp(0, 1):viewAs(target_img)):clamp(0, 1)
  elseif mode == 'hsl-polar-full' then
    -- Hue scaling in polar coordinates
    local s_hsl = image.rgb2hsl(source_img):view(source_img:size(1), source_img[1]:nElement())
    local t_hsl = image.rgb2hsl(target_img):view(target_img:size(1), target_img[1]:nElement())

    local sMean, sVar, tMean, tVar = torch.Tensor(3), torch.Tensor(3), torch.Tensor(3), torch.Tensor(3)
    sMean[2], sMean[3] = s_hsl[2]:mean(), s_hsl[3]:mean()
    tMean[2], tMean[3] = t_hsl[2]:mean(), t_hsl[3]:mean()
    sVar[2], sVar[3] = torch.var(s_hsl[2], 1, true)[1] + eps, torch.var(s_hsl[3], 1, true)[1] + eps
    tVar[2], tVar[3] = torch.var(t_hsl[2], 1, true)[1] + eps, torch.var(t_hsl[3], 1, true)[1] + eps

    -- Averaging hue in HSL makes significant wrong shift, taking mean hue from averaged RGB
    sMean[1] = image.rgb2hsl(torch.mean(source_img, 3):mean(2)):squeeze()[1]
    tMean[1] = image.rgb2hsl(torch.mean(target_img, 3):mean(2)):squeeze()[1]

    -- Finding source hue deltas
    local hd1 = s_hsl[1] - sMean[1]
    local hd2 = hd1 + 1
    local hd3 = hd1 - 1
    local hm = torch.lt(torch.abs(hd2), torch.abs(hd1))
    hd1[hm] = hd2[hm]
    hm = torch.lt(torch.abs(hd3), torch.abs(hd1))
    hd1[hm] = hd3[hm]
    s_hsl[1] = hd1   -- original hue can still be restored as (s_hsl[1] + sMean[1]):remainder(1)
    -- Same for target
    hd1 = t_hsl[1] - tMean[1]
    hd2 = hd1 + 1
    hd3 = hd1 - 1
    hm = torch.lt(torch.abs(hd2), torch.abs(hd1))
    hd1[hm] = hd2[hm]
    hm = torch.lt(torch.abs(hd3), torch.abs(hd1))
    hd1[hm] = hd3[hm]
    t_hsl[1] = hd1

    -- Hue variance
    sVar[1] = torch.abs(s_hsl[1]):pow(2):mean() + eps
    tVar[1] = torch.abs(t_hsl[1]):pow(2):mean() + eps

    -- Soft limit
    local recolor_strength_lim = params.recolor_strength
    local recolor_strength_sign; if recolor_strength_lim < 0 then recolor_strength_sign = -1 else recolor_strength_sign = 1 end
    recolor_strength_lim = (math.abs(recolor_strength_lim) ^ (1/1.11)) * recolor_strength_sign
    -- Scaling hue, "ultraviolet" and "infrared" regions are cut off
    t_hsl[1]:mul((sVar[1] / tVar[1]) ^ (params.recolor_strength / 8)):clamp(-0.5, 0.5):add(tMean[1] + (sMean[1] - tMean[1]) * recolor_strength_lim):remainder(1)
    -- Scaling saturation / lightness
--    if recolor_strength_lim > 1 then recolor_strength_lim = 1 end
    t_hsl[2]:add(-tMean[2]):mul((sVar[2] / tVar[2]) ^  params.recolor_strength / 2 ):add(tMean[2] + (sMean[2] - tMean[2]) * recolor_strength_lim)
    t_hsl[3]:add(-tMean[3]):mul((sVar[3] / tVar[3]) ^ (params.recolor_strength / 4)):add(tMean[3] + (sMean[3] - tMean[3]) * recolor_strength_lim)

    return image.hsl2rgb(t_hsl:clamp(0, 1):viewAs(target_img)):clamp(0, 1)
  elseif mode == 'hsl-polar' then
    -- Hue scaling in polar coordinates, saturation-weighted
    local s_hsl = image.rgb2hsl(source_img):view(source_img:size(1), source_img[1]:nElement())
    local t_hsl = image.rgb2hsl(target_img):view(target_img:size(1), target_img[1]:nElement())

    local sMean, sVar = s_hsl:mean(2):squeeze(), torch.Tensor(3)
    local tMean, tVar = t_hsl:mean(2):squeeze(), torch.Tensor(3)
    sVar[2], sVar[3] = torch.var(s_hsl[2], 1, true)[1] + eps, torch.var(s_hsl[3], 1, true)[1] + eps
    tVar[2], tVar[3] = torch.var(t_hsl[2], 1, true)[1] + eps, torch.var(t_hsl[3], 1, true)[1] + eps

    -- Finding source hue deltas
    local hd1 = s_hsl[1] - sMean[1]
    local hd2 = hd1 + 1
    local hd3 = hd1 - 1
    local hm = torch.lt(torch.abs(hd2), torch.abs(hd1))
    hd1[hm] = hd2[hm]
    hm = torch.lt(torch.abs(hd3), torch.abs(hd1))
    hd1[hm] = hd3[hm]
    s_hsl[1] = hd1   -- original hue can still be restored as (s_hsl[1] + sMean[1]):remainder(1)
    -- Same for target
    hd1 = t_hsl[1] - tMean[1]
    hd2 = hd1 + 1
    hd3 = hd1 - 1
    hm = torch.lt(torch.abs(hd2), torch.abs(hd1))
    hd1[hm] = hd2[hm]
    hm = torch.lt(torch.abs(hd3), torch.abs(hd1))
    hd1[hm] = hd3[hm]
    t_hsl[1] = hd1

    -- Hue variance, saturation-weighted
    sVar[1] = torch.abs(torch.cmul(s_hsl[1], s_hsl[2])):pow(0.75):sum() / torch.sum(s_hsl[2]) + eps
    tVar[1] = torch.abs(torch.cmul(t_hsl[1], t_hsl[2])):pow(0.75):sum() / torch.sum(t_hsl[2]) + eps

    -- Soft limit
    local recolor_strength_lim = params.recolor_strength
    local recolor_strength_sign; if recolor_strength_lim < 0 then recolor_strength_sign = -1 else recolor_strength_sign = 1 end
    recolor_strength_lim = (math.abs(recolor_strength_lim) ^ (1/1.11)) * recolor_strength_sign
    -- Scaling hue, "ultraviolet" and "infrared" regions are cut off
    t_hsl[1]:mul((sVar[1] / tVar[1]) ^ (params.recolor_strength / 0.75)):clamp(-0.5, 0.5):add(tMean[1] + (sMean[1] - tMean[1]) * recolor_strength_lim):remainder(1)
    -- Scaling saturation / lightness
    -- if recolor_strength_lim > 1 then recolor_strength_lim = 1 end
    t_hsl[2]:add(-tMean[2]):mul((sVar[2] / tVar[2]) ^  params.recolor_strength / 2 ):add(tMean[2] + (sMean[2] - tMean[2]) * recolor_strength_lim)
    t_hsl[3]:add(-tMean[3]):mul((sVar[3] / tVar[3]) ^ (params.recolor_strength / 4)):add(tMean[3] + (sMean[3] - tMean[3]) * recolor_strength_lim)

    return image.hsl2rgb(t_hsl:clamp(0, 1):viewAs(target_img)):clamp(0, 1)
  elseif mode == 'lab-rgb' then
    local s_lab = image.rgb2lab(source_img)  -- -100...100 range?
    local t_lab = image.rgb2lab(target_img)
    local sMean = torch.Tensor({torch.mean(s_lab[1]), torch.mean(s_lab[2]), torch.mean(s_lab[3])}):view(3,1,1)
    local sStd = torch.Tensor( {torch.std( s_lab[1]), torch.std( s_lab[2]), torch.std( s_lab[3])}):view(3,1,1)
    local tMean = torch.Tensor({torch.mean(t_lab[1]), torch.mean(t_lab[2]), torch.mean(t_lab[3])}):view(3,1,1)
    local tStd = torch.Tensor( {torch.std( t_lab[1]), torch.std( t_lab[2]), torch.std( t_lab[3])}):view(3,1,1)
    local tCol = t_lab - tMean:expandAs(t_lab)
    tCol = tCol:cmul(sStd:expandAs(tCol)):cdiv(tStd:expandAs(tCol)) + sMean:expandAs(tCol)
    tCol_lab = image.lab2rgb(tCol)
    sMean = torch.Tensor({torch.mean(source_img[1]), torch.mean(source_img[2]), torch.mean(source_img[3])}):view(3,1,1)
    sStd = torch.Tensor( {torch.std( source_img[1]), torch.std( source_img[2]), torch.std( source_img[3])}):view(3,1,1)
    tMean = torch.Tensor({torch.mean(target_img[1]), torch.mean(target_img[2]), torch.mean(target_img[3])}):view(3,1,1)
    tStd = torch.Tensor( {torch.std( target_img[1]), torch.std( target_img[2]), torch.std( target_img[3])}):view(3,1,1)
    tCol = target_img - tMean:expandAs(target_img)
    tCol = tCol:cmul(sStd:expandAs(tCol)):cdiv(tStd:expandAs(tCol)) + sMean:expandAs(tCol)

    --return ((tCol_lab + tCol) / 2):clamp(0, 1)
    return torch.cmul(tCol_lab, tCol):sqrt():cmul(tCol):sqrt():clamp(0, 1)
  elseif mode == 'idt' then
    -- Direct reimplementation in Torch of https://github.com/frcs/colour-transfer, (c) F. Pitie 2007.
    -- Modified (can Torch divide 2 non-square matrices?), but seems to work.

    local nb_iterations = 10 -- calculation time is proportional

    local D0 = target_img:view(target_img:size(1), target_img[1]:nElement())
    local D1 = source_img:view(source_img:size(1), source_img[1]:nElement())

    print('Building a sequence of (almost) random projections.')
    local R = torch.Tensor(nb_iterations, 3, 3)
    R[1] = torch.Tensor({{ 1.0,  0.0,  0.0},
                         { 0.0,  1.0,  0.0},
                         { 0.0,  0.0,  1.0}})
--  Temporarily removed.
--  R[2] = torch.Tensor({{ 2/3,  2/3, -1/3},
--                       { 2/3, -1/3,  2/3},
--                       {-1/3,  2/3,  2/3}})
    for i = 2, nb_iterations do
      R[i]   = R[1] * ml_orth(torch.rand(3, 3))
    end

    print('Probability density function transfer.')
    return pdf_transfer(D0, D1, R, params.recolor_strength):viewAs(target_img)
  elseif mode == 'idt-mean' then
    -- Direct reimplementation in Torch of https://github.com/frcs/colour-transfer, (c) F. Pitie 2007.
    -- Modified (can Torch divide 2 non-square matrices?), but seems to work.
    -- With "mean-weighted" linear interpolation function.

    local nb_iterations = 10 -- calculation time is proportional

    local D0 = target_img:view(target_img:size(1), target_img[1]:nElement())
    local D1 = source_img:view(source_img:size(1), source_img[1]:nElement())

    local R = torch.Tensor(3, 3)
    R = torch.Tensor({{ 1.0,  0.0,  0.0},
                      { 0.0,  1.0,  0.0},
                      { 0.0,  0.0,  1.0}})

    print('Probability density function transfer (mean-weighted).')
    return pdf_transfer_mean_weighted(D0, D1, R, nb_iterations, params.recolor_strength):viewAs(target_img)
  elseif mode == 'rgb-hist' then
    -- Weighted by histogram scaling of RGB channels.
    -- Inspired by https://github.com/frcs/colour-transfer

    local iterations = 3   -- More iterations coloring harder, but computation time is almost squared times longer.

    print(string.format('Histogram-weighted RGB color transfer.'))
    local eps = 1e-10
    local ch_s = source_img:size(1)
    local lin_s, lin_d = source_img:view(ch_s, source_img[1]:nElement()), target_img:view(ch_s, target_img[1]:nElement())

    -- Initialazing with image, scaled by standard deviation.
    local mean_s, mean_d = lin_s:mean(2):view(ch_s,1,1), lin_d:mean(2):view(ch_s,1,1)
    local std_s, std_d   = lin_s:std(2),  lin_d:std(2)
    local dr = (target_img - mean_d:expandAs(target_img)):cmul(torch.cdiv(std_s, std_d):view(ch_s,1,1):expandAs(target_img)):add(mean_s:expandAs(target_img))
    --local dr = torch.Tensor(target_img:size()):zero()

    for hist_points = 1, iterations do
      print(string.format('Iteration %d / %d', hist_points, iterations))
      local lin_r = torch.Tensor(lin_d:size())
      for chan_i = 1, ch_s do
        --print(string.format('Channel %d / %d', chan_i, ch_s))
        lin_r[chan_i] = reshape_histogram(lin_s[chan_i], lin_d[chan_i], hist_points * 3)
      end

      -- 1) normal sum, faster, must be zero at start, only final result is valid
      --dr:add(lin_r:viewAs(target_img):div(iterations))
      -- 2) normal sum, every iteration is valid, initial state does not matter
      --dr:mul((hist_points - 1) / hist_points):add(lin_r:viewAs(target_img):div(hist_points))
      -- 3) fading sum, must be initialized with std at start
      --dr:add(lin_r:viewAs(target_img):div(hist_points)):div((hist_points + 1) / hist_points)
      -- 4) fading sum, x^(1/x...x/x), rasing smoothness
      --local add_part = hist_points ^ (1 / (hist_points ^ (1 / (hist_points + 2))))
      --dr:add(lin_r:viewAs(target_img):div(add_part)):div((add_part + 1) / add_part)
      -- 5) fading sum, x^(1/x...x^2), exponential smoothness                                                                -- v Raise for smoother colors.
      local add_part = hist_points ^ (( (hist_points ^ (1 - (2 / hist_points))) / (hist_points ^ ((4 / hist_points) - 1)) ) ^ (1/3) / 2)
      dr:add(lin_r:viewAs(target_img):div(add_part)):div((add_part + 1) / add_part)
    end

    return dr:clamp(0, 1)
  end

  -- from Leon Gatys's code: https://github.com/leongatys/NeuralImageSynthesis/blob/master/ExampleNotebooks/ColourControl.ipynb
  -- and ProGamerGov's code: https://github.com/jcjohnson/neural-style/issues/376, https://github.com/ProGamerGov/Neural-Tools
  local eyem = torch.eye(source_img:size(1)):mul(eps)

  local mu_s = torch.mean(source_img, 3):mean(2)
  local s = source_img - mu_s:expandAs(source_img)
  s = s:view(s:size(1), s[1]:nElement())
  local Cs = s * s:t() / s:size(2) + eyem

  local mu_t = torch.mean(target_img, 3):mean(2)
  local t = target_img - mu_t:expandAs(target_img)
  t = t:view(t:size(1), t[1]:nElement())
  local Ct = t * t:t() / t:size(2) + eyem

  local ts
  if mode == 'chol' then
    local chol_s = torch.potrf(Cs, 'L')
    local chol_t = torch.potrf(Ct, 'L')
    ts = chol_s * torch.inverse(chol_t)
  elseif mode == 'pca' then
    local eva_t, eve_t = torch.symeig(Ct, 'V', 'L')
    local Qt = eve_t * torch.diag(eva_t):sqrt() * eve_t:t()
    local eva_s, eve_s = torch.symeig(Cs, 'V', 'L')
    local Qs = eve_s * torch.diag(eva_s):sqrt() * eve_s:t()
    ts = Qs * torch.inverse(Qt)
  elseif mode == 'mkl' then
    -- https://github.com/frcs/colour-transfer
    -- https://github.com/mdfirman/python_colour_transfer
    --[[
    [Ua,Da2] = eig(A);
    Da2 = diag(Da2);
    Da2(Da2<0) = 0;
    Da = diag(sqrt(Da2 + eps));
    C = Da*Ua'*B*Ua*Da;
    [Uc,Dc2] = eig(C);
    Dc2 = diag(Dc2);
    Dc2(Dc2<0) = 0;
    Dc = diag(sqrt(Dc2 + eps));
    Da_inv = diag(1./(diag(Da)));
    T = Ua*Da_inv*Uc*Dc*Uc'*Da_inv*Ua';
    --]]

    eps = eps ^ 2
    local Da2, Ua = torch.symeig(Ct, 'V', 'U')
    Da2[torch.lt(Da2, 0)] = 0
    local Da = torch.diag((Da2 + eps):sqrt())
    local C = Da * Ua:t() * Cs * Ua * Da
    local Dc2, Uc = torch.symeig(C, 'V', 'U')
    Dc2[torch.lt(Dc2, 0)] = 0
    local Dc = torch.diag((Dc2 + eps):sqrt())
    local Da_inv = torch.inverse(Da)
    ts = Ua * Da_inv * Uc * Dc * Uc:t() * Da_inv * Ua:t()
  elseif mode == 'sym' then
    local eva_t, eve_t = torch.symeig(Ct, 'V', 'L')
    local Qt = eve_t * torch.diag(eva_t):sqrt() * eve_t:t()
    local Qt_Cs_Qt = Qt * Cs * Qt
    local eva_QtCsQt, eve_QtCsQt = torch.symeig(Qt_Cs_Qt, 'V', 'L')
    local QtCsQt = eve_QtCsQt * torch.diag(eva_QtCsQt):sqrt() * eve_QtCsQt:t()
    local iQt = torch.inverse(Qt)
    ts = iQt * QtCsQt * iQt
  elseif mode == 'chol-pca' then
    local chol_s = torch.potrf(Cs, 'L')
    local chol_t = torch.potrf(Ct, 'L')
    local ts_chol = chol_s * torch.inverse(chol_t)

    local eva_t, eve_t = torch.symeig(Ct, 'V', 'L')
    local Qt = eve_t * torch.diag(eva_t):sqrt() * eve_t:t()
    local eva_s, eve_s = torch.symeig(Cs, 'V', 'L')
    local Qs = eve_s * torch.diag(eva_s):sqrt() * eve_s:t()
    local ts_pca = Qs * torch.inverse(Qt)

    ts = (ts_chol + ts_pca) / 2
  elseif mode == 'chol-sym' then
    local chol_s = torch.potrf(Cs, 'L')
    local chol_t = torch.potrf(Ct, 'L')
    local ts_chol = chol_s * torch.inverse(chol_t)

    local eva_t, eve_t = torch.symeig(Ct, 'V', 'L')
    local Qt = eve_t * torch.diag(eva_t):sqrt() * eve_t:t()
    local Qt_Cs_Qt = Qt * Cs * Qt
    local eva_QtCsQt, eve_QtCsQt = torch.symeig(Qt_Cs_Qt, 'V', 'L')
    local QtCsQt = eve_QtCsQt * torch.diag(eva_QtCsQt):sqrt() * eve_QtCsQt:t()
    local iQt = torch.inverse(Qt)
    local ts_sym = iQt * QtCsQt * iQt

    ts = (ts_chol + ts_sym) / 2
  elseif mode == 'exp1' then
    local chol_s = torch.potrf(Cs, 'L')
    local chol_t = torch.potrf(Ct, 'L')
    local ts_chol = chol_s * torch.inverse(chol_t)

    local eva_t, eve_t = torch.symeig(Ct, 'V', 'L')
    local Qt = eve_t * torch.diag(eva_t):sqrt() * eve_t:t()
    local Qt_Cs_Qt = Qt * Cs * Qt
    local eva_QtCsQt, eve_QtCsQt = torch.symeig(Qt_Cs_Qt, 'V', 'L')
    local QtCsQt = eve_QtCsQt * torch.diag(eva_QtCsQt):sqrt() * eve_QtCsQt:t()
    local iQt = torch.inverse(Qt)
    local ts_sym = iQt * QtCsQt * iQt

    --ts = (ts_chol + ts_sym):abs():cmul((ts_chol - ts_sym):abs()):sqrt()
    ts = ( (ts_chol + ts_sym):abs():cmul((ts_chol - ts_sym):abs()):sqrt()  + ts_chol + ts_sym ) / 3
  else
    error('Unknown color matching mode. Stop.')
  end

  local matched_img = (ts * t):viewAs(target_img) + mu_s:expandAs(target_img)
  -- matched_img = image.minmax{tensor=matched_img, min=0, max=1}
  return matched_img:clamp(0, 1)
end


main(params)
