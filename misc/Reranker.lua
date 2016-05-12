require 'torch'
require 'nn'
require 'image'
require 'loadcaffe'
w2v = require 'word2vec.torch/w2vutils'

local Reranker, parent = torch.class('nn.Reranker', 'nn.Module')

function Reranker:__init(params, proto_file, model_file, word2vec_karpathy, vocab, vocab_size)
  parent.__init(self)

  self.params = params
  self.vocab_size = vocab_size
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
  local seq_length = beams[1][1].seq:size(1)
  local seq = torch.LongTensor(seq_length, batchSize)
  local seqLogprobs = torch.FloatTensor(seq_length, batchSize)

  for k = 1, batchSize do
    local pred = self.cnn:forward(images[k])
    if self.params.rerank == 'P1+P2' then
      self:weightedsum_strategy(k, pred, beams, seq, seqLogprobs, seq_length)
    elseif self.params.rerank == 'rank1+rank2' then
      self:addrank_strategy(k, pred, beams, seq, seqLogprobs, seq_length)
    elseif self.params.rerank == 'countHighProbObj' then
      self:counthighprobobject_strategy(k, pred, beams, seq, seqLogprobs, seq_length)
    else
      assert(false, 'invalid argument: -rerank')
    end
    
  end

  return seq, seqLogprobs
end

function Reranker:counthighprobobject_strategy(k, pred, beams, seq, seqLogprobs, seq_length)
    local high_prob_obj_cnts = torch.Tensor(#beams[k]):fill(0)
    local HIGH_PROB_THRESHOLD = self.params.high_prob_thres
    for i = 1, #beams[k] do -- beams[k][i] is current sentence
      local sent = beams[k][i].seq     
      local sent_str = ''
      high_prob_obj_cnts[i] = 0
      local sent_END = sent:size(1)
      for j = 1, sent:size(1) do -- beams[k][i].seq[j] is current word
        if sent[j] > self.vocab_size then -- assume this is the END token
          sent_END = j
        end 
        local NN_idx = self:NN_in_ImgNet(self.vocab[tostring(sent[j])]) -- indices in the 1000 ImageNet class nouns
        local i_max_prob, max_prob = -1, 0
        for k = 1,#NN_idx do if pred[NN_idx[k]] > max_prob then max_prob=pred[NN_idx[k]]; i_max_prob = NN_idx[k] end end
        
        local near_word_in_ImgNet = (i_max_prob > 0) and (string.format('%s%f', table.concat(self.synset_words[i_max_prob]), pred[i_max_prob])) or ('')
        if i_max_prob > 0 and pred[i_max_prob] > HIGH_PROB_THRESHOLD and j < sent_END then -- don't count words after END token(j>sent_END)
          high_prob_obj_cnts[i] = high_prob_obj_cnts[i] + 1
        end
        sent_str = sent_str .. ' ' .. (self.vocab[tostring(sent[j])]==nil and 'nil'..sent[j] or  (self.vocab[tostring(sent[j])] .. '('..near_word_in_ImgNet..')' ) )
      end
      beams[k][i].sent_str = sent_str
    end
    local sorted_cnt,idx = torch.sort(high_prob_obj_cnts,1,true) -- "true" =  descending order
    for j=1, #beams[k]  do beams[k][j].rank1 = j end
    local rank2 = 1
    for j=1, #beams[k]  do  -- assign rank2 to each sentence
      if j>1 and sorted_cnt[j] ~= sorted_cnt[j-1] then rank2 = j end
      beams[k][idx[j]].rank2 = rank2
    end
    for j=1, #beams[k]  do beams[k][j].rank = beams[k][j].rank1 + beams[k][j].rank2 end
    
    if self.params.reranker_debug_info == 1 then
      for i=1, #beams[k] do print(beams[k][i].sent_str, string.format('logP1:%.0f highprob_cnt:%d rank(1,2,1+2)=%d,%d,%d', beams[k][i].p, high_prob_obj_cnts[i], beams[k][i].rank1, beams[k][i].rank2, beams[k][i].rank )) end
    end
    local function compare(a,b) return a.rank < b.rank end -- used downstream
    table.sort(beams[k], compare)
    seq[{ {}, k }] = beams[k][1].seq
    seqLogprobs[{ {}, k }] = beams[k][1].logps
    
    if self.params.reranker_debug_info == 1 then
      print('best new rank is', beams[k][1].rank)
    end
end

function Reranker:addrank_strategy(k, pred, beams, seq, seqLogprobs, seq_length)
    local logP2s = torch.Tensor(#beams[k]):fill(0)
    for i = 1, #beams[k] do -- beams[k][i] is current sentence
      local logP2 = 0
      local sent = beams[k][i].seq     
      local sent_str = ''
      for j = 1, sent:size(1) do -- beams[k][i].seq[j] is current word
        local NN_idx = self:NN_in_ImgNet(self.vocab[tostring(sent[j])]) -- indices in the 1000 ImageNet class nouns
        local i_max_prob, max_prob = -1, 0
        for k = 1,#NN_idx do if pred[NN_idx[k]] > max_prob then max_prob=pred[NN_idx[k]]; i_max_prob = NN_idx[k] end end
        
        local near_word_in_ImgNet = (i_max_prob > 0) and (string.format('%s%.2f', table.concat(self.synset_words[i_max_prob]), pred[i_max_prob])) or ('')
        if i_max_prob > 0 then
          logP2 = logP2 + torch.log(pred[i_max_prob])
        end
        sent_str = sent_str .. ' ' .. (self.vocab[tostring(sent[j])]==nil and 'nil'..sent[j] or  (self.vocab[tostring(sent[j])] .. '('..near_word_in_ImgNet..')' ) )
      end
      beams[k][i].sent_str = sent_str
      beams[k][i].p2 = logP2
      logP2s[i] = logP2
    end
    local sortedP2,idx = torch.sort(logP2s,1,true) -- "true" =  descending order
    for j=1, #beams[k]  do beams[k][j].rank1 = j end
    local rank2 = 1
    for j=1, #beams[k]  do  -- assign rank2 to each sentence
      if j>1 and sortedP2[j] ~= sortedP2[j-1] then rank2 = j end
      beams[k][idx[j]].rank2 = rank2
    end
    for j=1, #beams[k]  do beams[k][j].rank = beams[k][j].rank1 + beams[k][j].rank2 end
    
    if self.params.reranker_debug_info == 1 then
      for i=1, #beams[k] do print(beams[k][i].sent_str, string.format('logP1:%.0f logP2:%.0f rank(1,2,1+2)=%d,%d,%d', beams[k][i].p, beams[k][i].p2, beams[k][i].rank1, beams[k][i].rank2, beams[k][i].rank )) end
    end
    local function compare(a,b) return a.rank < b.rank end -- used downstream
    table.sort(beams[k], compare)
    seq[{ {}, k }] = beams[k][1].seq
    seqLogprobs[{ {}, k }] = beams[k][1].logps
    
    if self.params.reranker_debug_info == 1 then
      print('best new rank is', beams[k][1].rank)
      print('logP2:')
      print(logP2s)
    end
end


function Reranker:weightedsum_strategy(k, pred, beams, seq, seqLogprobs, seq_length)
    for i = 1, #beams[k] do
      local logP2 = 0
      local sent = beams[k][i].seq     
      local n = 0
      local sent_str = ''
      for j = 1, sent:size(1) do
        local NN_idx = self:NN_in_ImgNet(self.vocab[tostring(sent[j])]) -- indices in the 1000 ImageNet class nouns
        local i_max_prob, max_prob = -1, 0
        for k = 1,#NN_idx do if pred[NN_idx[k]] > max_prob then max_prob=pred[NN_idx[k]]; i_max_prob = NN_idx[k] end end
        
        local near_word_in_ImgNet = (i_max_prob > 0) and (string.format('%s%.2f', table.concat(self.synset_words[i_max_prob]), pred[i_max_prob])) or ('')
        if i_max_prob > 0 then
          logP2 = logP2 + torch.log(pred[i_max_prob])
        end
        
        n = n + 1
        sent_str = sent_str .. ' ' .. (self.vocab[tostring(sent[j])]==nil and 'nil'..sent[j] or  (self.vocab[tostring(sent[j])] .. '('..near_word_in_ImgNet..')' ) )
        
      end
      alpha = 0.5
      -- I found beams[k][i].p is simply the sum of all $seq_length log probabilities of every word, and is used as the key of sorting in function sample_beam(LanguageModel.lua).
      local P = torch.exp(beams[k][i].p/seq_length) * alpha + torch.exp(logP2/n) * (1-alpha)
      if self.params.reranker_debug_info == 1 then print(sent_str, string.format('logP1/16:%.2f logP2/n:%.2f P:%.2f', beams[k][i].p/16, logP2/n, P)) end
      beams[k][i].p_rerank = P
    end

    local function compare(a,b) return a.p_rerank > b.p_rerank end -- used downstream
    table.sort(beams[k], compare)
    seq[{ {}, k }] = beams[k][1].seq
    seqLogprobs[{ {}, k }] = beams[k][1].logps

    local _a,_idx=torch.max(pred,1)
    if self.params.reranker_debug_info == 1 then print('maximum P is', beams[k][1].p_rerank, string.format('ImageNet pred is %.2f %s', _a[1], table.concat(self.synset_words[_idx[1]]))) end
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

-- based on tiny experiment results, can try return k-NN (k=5), and we will have 5 probabilities, and take their max. This might improve performance.
function Reranker:NN_in_ImgNet(word_str) -- nearest neighbour in ImageNet 1000 class nouns, returns -1 if the minimum distance is greater than a threshold (too far away)
  -- similarity mesure: dotProduct 
  local THRESHOLD = self.params.nn_similar_thres
  local k = 5 -- the k in 'k-NN'
  if word_str == nil then word_str = 'nil' end
  
  if _cache_NN_query == nil then
    _cache_NN_query={} 
  else
    if _cache_NN_query[word_str] then 
      return _cache_NN_query[word_str]
    end
  end
  
  local answer = {}
  local vec = self:get_words_vec({word_str})
  if torch.norm(vec-w2v.UNK) < 1e-5 then
    answer = {}
  else
    local dotprod = torch.mv(self.synset_vecs ,vec)
    dotprod , idx = torch.sort(dotprod,1,true)
    for i=1,k do
      if dotprod[i] < THRESHOLD then
        break
      else
        answer[i] = idx[i]
      end
    end
  end
  
  --deal with plural words 
  if self.params.consider_plural == 'Plu' and string.sub(word_str,-1,-1)=='s' then
    TableConcat(answer, self:NN_in_ImgNet(string.sub(word_str,1,-2)))
  end
  
  _cache_NN_query[word_str] = answer
  return answer
end

function TableConcat(t1,t2)
    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
end