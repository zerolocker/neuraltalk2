require 'torch'
require 'nn'
require 'image'
require 'loadcaffe'

local Reranker, parent = torch.class('nn.Reranker', 'nn.Module')
function Reranker:__init(params, proto_file, model_file)
  parent.__init(self)

  if params.gpuid >= 0 then
  if params.backend ~= 'clnn' then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(params.gpu + 1)
  else
    require 'clnn'
    require 'cltorch'
    cltorch.setDevice(params.gpu + 1)
  end
  else
    params.backend = 'nn'
  end

  if params.backend == 'cudnn' then
    require 'cudnn'
    if params.cudnn_autotune then
      cudnn.benchmark = true
    end
    cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
  end
  
  local loadcaffe_backend = params.backend
  if params.backend == 'clnn' then loadcaffe_backend = 'nn' end
  self.cnn = loadcaffe.load(proto_file, model_file, loadcaffe_backend):float()
  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      cnn:cuda()
    else
      cnn:cl()
    end
  end
end
