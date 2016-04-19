require 'torch'
require 'nn'
require 'image'
require 'loadcaffe'

local Reranker, parent = torch.class('nn.Reranker', 'nn.Module')

function Reranker:__init(params, proto_file, model_file)
  parent.__init(self)

  -- load VGG
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
  self.cnn = loadcaffe.load(proto_file, model_file, loadcaffe_backend):float()
  if params.gpuid >= 0 then
    if params.backend ~= 'clnn' then
      self.cnn:cuda()
    else
      self.cnn:cl()
    end
  end

  -- load ILSVRC12 synset_word
  self.synset_words = {}
  for line in io.lines'models/synset_words.txt' do
    table.insert(self.synset_words, line:sub(11))
  end

end


function Reranker:rank(beams, vocab, images)
  batchSize = #beams
  seq = torch.LongTensor(beams[1][1].seq:size(1), batchSize)

  -- TODO: implement reranking
  for k = 1, batchSize do
    pred = self.cnn:forward(images[k])

    for i = 1, #beams[k] do
      candidate = beams[k][i].seq
      for j = 1, candidate:size(1) do
        --print(candidate[j])
        --print(vocab[candidate[j]])
      end
    end

    seq[{ {}, k }] = beams[k][1].seq
  end

  return seq
end
