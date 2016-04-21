require 'torch'
require 'nn'
require 'image'
require 'loadcaffe'

local Reranker, parent = torch.class('nn.Reranker', 'nn.Module')

function Reranker:__init(params, proto_file, model_file, word2vec, vocab, vocab_size)
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

  -- set up vocabuary and inverse vocabulary
  self.vocab = vocab
  self.inv_vocab = {}
  for i = 1, vocab_size do
    self.inv_vocab[vocab[tostring(i)]] = i
  end

  -- TODO:
  -- currently the word2vec function is the learned embedding of Karpathy's alignment model
  -- may use Google's word2vec trained on other corpus(billions of words)
  -- but which one is better?
  self.word2vec = word2vec

  -- load ILSVRC12 synset_word
  self.synset_words = {}
  for line in io.lines'models/synset_words.txt' do
    line = line:sub(11)
    phrases = {}
    for phrase in string.gmatch(line, '([^,]+)') do
      table.insert(phrases, phrase)
    end
    table.insert(self.synset_words, phrases)
  end
  --print(self.synset_words)

  -- TODO: 
  -- map ILSVRC12 synset_word to vector representation
  -- not a perfect method
  self.synset_vecs = {}
  local l = 0
  for line in io.lines'models/synset_words.txt' do
    line = line:sub(11)
    words = {}
    for phrase in string.gmatch(line, '([^,]+)') do
      for word in string.gmatch(phrase, '([^ ]+)') do
        if (self.inv_vocab[word] == nil) then
--          print(word .. ' is out of vocabulary')
        else
          table.insert(words, self.inv_vocab[word])
        end
      end
    end

    if (#words > 0) then
      local vecs = self.word2vec:forward(torch.Tensor(words))
      self.synset_vecs[l] = vecs:float() -- entries will have same size if using cudaTensor, why?
    end
    l = l + 1
  end
end


function Reranker:rank(beams, images)
  local batchSize = #beams
  local seq = torch.LongTensor(beams[1][1].seq:size(1), batchSize)

  -- TODO: implement reranking
  for k = 1, batchSize do
    local pred = self.cnn:forward(images[k])
    for i = 1, #beams[k] do
      local logP2 = 0
      local sent = beams[k][i].seq     
      local sentVec = self.word2vec:forward(sent)
      local n = 0
      local sent_str = ''
      for j = 1, sent:size(1) do
        local label = 1 --nearestNeighbor(sentVec[j], self.synset_vecs)
        logP2 = logP2 + torch.log(pred[label])
        
        n = n + 1
 	sent_str = sent_str .. ' ' .. (self.vocab[tostring(sent[j])]==nil and 'nil'..sent[j] or self.vocab[tostring(sent[j])])
      end
      print(sent_str,'logProb: ' .. beams[k][i].p)
      alpha = 0.5
      local P = torch.exp(beams[k][i].p) * alpha + torch.pow(torch.exp(logP2),n) * (1-alpha)
    end

    seq[{ {}, k }] = beams[k][1].seq
  end

  return seq
end
