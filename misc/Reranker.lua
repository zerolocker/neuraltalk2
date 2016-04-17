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
      cutorch.setDevice(params.gpuid + 1)
    else
      require 'clnn'
      require 'cltorch'
      cltorch.setDevice(params.gpuid + 1)
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
  local cnn = loadcaffe.load(proto_file, model_file, loadcaffe_backend):float()
  if params.gpuid >= 0 then
    if params.backend ~= 'clnn' then
      cnn:cuda()
    else
      cnn:cl()
    end
  end
end


function Reranker:rank(beams)
  batchSize = #beams
  seq = torch.LongTensor(beams[1][1].seq:size(1), batchSize)

  -- TODO: implement reranking
  for k = 1, batchSize do
    seq[{ {}, k }] = beams[k][1].seq
  end

  return seq
end
