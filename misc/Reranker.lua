require 'torch'
require 'nn'
require 'image'
require 'loadcaffe'
w2v = require 'word2vec.torch/w2vutils'

local Reranker, parent = torch.class('nn.Reranker', 'nn.Module')

function Reranker:__init(params, proto_file, model_file, word2vec_karpathy, vocab, vocab_size)
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

  -- set up the vocabuary of dataset(not ImageNet) and inverse vocabulary
  self.vocab = vocab
  self.inv_vocab = {}
  for i = 1, vocab_size do
    self.inv_vocab[vocab[tostring(i)]] = i
  end

  -- load ILSVRC12 synset_word and transform into vec
  self.synset_words = {}
  self.synset_vecs = torch.Tensor(1000,w2v.M:size(2)):fill(0)
  self.unk = {}
  i = 1
  for line in io.lines'models/synset_words.txt' do
    local words = {}
    for word in string.gmatch(line, '([^|]+)') do
      table.insert(words, tostring(word:gsub('[ -]','_'))) -- replace space with _ (in accordance to word2vec)
    end
    table.insert(self.synset_words, words)
    self.synset_vecs[i], self.unk[i] = self:get_words_vec(words)
    i = i + 1
  end
  true_cnt=0;for i,b in ipairs(self.unk) do if b==true then true_cnt=true_cnt+1 end end
  -- print('OOV word cnt in ImageNet 1000 classes:', truecnt)
end


function Reranker:rank(beams, images)
  local batchSize = #beams
  local seq = torch.LongTensor(beams[1][1].seq:size(1), batchSize)


  for k = 1, batchSize do
    local pred = self.cnn:forward(images[k])
    for i = 1, #beams[k] do
      local logP2 = 0
      local sent = beams[k][i].seq     
      local n = 0
      local sent_str = ''
      for j = 1, sent:size(1) do
        local word_idx = self:NN_in_ImgNet(self.vocab[tostring(sent[j])]) -- index in the 1000 ImageNet class nouns
        local near_word_in_ImgNet = (word_idx > 0) and (table.concat(self.synset_words[word_idx])) or ('')
        logP2 = logP2 + torch.log(pred[word_idx])
        
        n = n + 1
        sent_str = sent_str .. ' ' .. (self.vocab[tostring(sent[j])]==nil and 'nil'..sent[j] or  (self.vocab[tostring(sent[j])] .. '('..near_word_in_ImgNet..')' ) )
        
      end
      print(sent_str,'logProb: ' .. beams[k][i].p)
      alpha = 0.5
      local P = torch.exp(beams[k][i].p) * alpha + torch.pow(torch.exp(logP2),n) * (1-alpha)
    end

    seq[{ {}, k }] = beams[k][1].seq
  end

  return seq
end

  -- return the average the vector of a list of words
  --  If a word is not in the dict(<UNK>), skip it. If all words are <UNK>, return the vector of <UNK>.
function Reranker:get_words_vec(words)
  local vec =  torch.Tensor(w2v.M:size(2)):fill(0)
  local cnt = 0
  for i, word in ipairs(words) do 
    local ind = w2v.w2vvocab[word]
    if ind == nil then -- if not found, try if the last word of this word phrase exists in word2vec vocabulary
      last_word_idx = string.find(word, '%a+$')
      if last_word_idx then
        ind = w2v.w2vvocab[string.sub(word,last_word_idx)]
      else
        -- still not found, skip
      end
    end
    if ind ~= nil then
      cnt = cnt + 1
      vec = vec + w2v.M[ind]
    end
  end
  if cnt == 0 then
    return w2v.UNK, true
  else
    return vec / cnt, false
  end
end

function Reranker:NN_in_ImgNet(word_str) -- nearest neighbour in ImageNet 1000 class nouns, returns -1 if the minimum distance is greater than a threshold (too far away)
  -- similarity mesure: -dotProduct 
  local THRESHOLD = 0.5
  local vec = self:get_words_vec({word_str}) -- (can use cache to speed it up) (possibly be the vector of <UNK>)
  if torch.norm(vec-w2v.UNK) < 1e-5 then
    return -1
  end
  local dotprod = torch.mv(self.synset_vecs ,vec)
  dotprod , idx = torch.sort(dotprod,1,true)
  if dotprod[1] < THRESHOLD then
    return -1
  else
    return idx[1]
  end
end
