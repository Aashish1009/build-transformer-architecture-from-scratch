# importing the requirements
import torch
import torch.nn as nn
import math

#class for the input embedding
class InputEmbedding(nn.Module): # first thing in transformer architecture

    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model #dimension of the token
        self.vocab_size = vocab_size #total no of token embeddings to train
        self.embedding = nn.Embedding(vocab_size,d_model)#parameter order matter

    def forward(self,x):
        #in the transformer paper they multiply the input embedding by sqrt(d_model) to scale the embedding
        return self.embedding(x)* math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module): # embeddings for the position of token so they distinguish the positions

    def __init__(self, d_model:int, seq_len:int,dropout:float):
        super().__init__()
        self.d_model = d_model #same model dim
        self.seq_len = seq_len #context length at the time of training
        self.dropout = nn.Dropout(dropout) #we apply dropout on top of both after adding input and positionembedding as sugeested in paper to avoid overfitting

        #create the positional embedding (seq_len,d_model)
        pe = torch.zeros(seq_len,d_model)

        #create the positional encoding (seq_len,1) required for calculating the formula of positional encoding as it is going to mulitly by div term based on odd or even under cos or sin
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)

        #create the div term which we muliply from the position 
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))

        #apply the positional encoding to the positional embedding
        pe[:,0::2]=torch.sin(position*div_term) # for every row we start from column 0 and take step 2 so 0,2,4
        pe[:,1::2]=torch.cos(position*div_term) # for every row we start from column 1 and take step 2 so 1,3,5

        #add a batch dimension to the positional embedding (1,seq_len,d_model)
        pe = pe.unsqueeze(0) #since we work in batches 

        #register the buffer of the positional embedding
        self.register_buffer("pe",pe) #register buffer means it is not trainable params  but we want to save the tensor weights for positional embedding

    def forward(self,x):

        #add the positional embedding to the input embedding (1,seq_len,d_model)
        x= x + (self.pe[:,:x.shape[1],:]).requires_grad_(False) # we want all batch(1) input size as not all sentence are of seq_len some are shorter and all dim .since it is not trainable we want grad false

        #apply the dropout
        return self.dropout(x) # we apply droupout after adding both


class LayerNormalization(nn.Module): #to maintain the data so that mean can be 0 and variance can be 1 

    def __init__(self,features:int,eps:float = 1e-6 ):
        super().__init__()
        self.eps = eps # small value to avoid division by zero
        #we also intoduce two param alpha and bias that intoduces some fluctuations in databecause having all the value between 0 and 1 may be too restrictive . the model will learn to tune this parameter to introduce fluctutaion when necessary 
        #fetaures tells for how many values we have to craete params like alpha and bias as its one per token
        self.alpha = nn.Parameter(torch.ones(features)) #we have ones for multilpicative
        self.bias = nn.Parameter(torch.zeros(features)) #we have 0 fro additive 

    def forward(self,x):

        #calculate the mean and standard deviation
        mean = x.mean(dim=-1,keepdim=True) #keepdim=True to maintain the same shape so we dont get error because of broadcasting
        std = x.std(dim=-1,keepdim=True)  #keepdim=True to maintain the same shape so we dont get error because of broadcasting

        #apply the layer normalization
        return self.alpha *(x-mean) / (std +self.eps)  + self.bias


class FeedForwardBlock(nn.Module): #the part which inlcude non linearity by expanding and then collapsing the dim

    def __init__(self,d_model:int,d_ff:int,dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)#(d_model,d_ff) expansion layer
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff,d_model)#(d_ff,d_model) compression layer

    def forward(self,x):

        #apply the feed forward block with relu activation and dropout
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttentionBlock(nn.Module):#the block responsible for attention work 
    def __init__(self,d_model:int,h:int,dropout:float ):
        super().__init__()
        self.d_model = d_model #dimension of the model
        self.h = h #number of heads
        #d_model should be divisible by h 
        assert d_model %h==0,"dmodel is not divisible by h"

        self.dropout = nn.Dropout(dropout)
        #we divide to get multiple dimesnions of the llm_heads responsibe for multiple perspective
        self.d_k = d_model//h #dimension of dk

        #define the linear layers for query, key, value and output
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)

        self.w_o = nn.Linear(d_model,d_model)


    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k = query.shape[-1] # dimension of dk as its the last dimension

        #multiply query with key by transposing the last two dimension seq_len,d_k to d_k,seq_len and dividing by sqrt(dk) to maintain normalization and standard deviation as suggested by paper
        attention_scores = query @ key.transpose(-2,-1) / math.sqrt(d_k) #(1,h,seq_len,seq_len)

        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)#replace all the value where mask is 0 to this so there softmax become 0

        attention_scores = attention_scores.softmax(dim=-1) #(1,h,seq_len,seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        #we return attenstion value for next layer and attention scores fro visualization 
        return (attention_scores @ value) , attention_scores


    def forward(self,q,k,v,mask):

        query = self.w_q(q) #(1,seq_len,d_model)
        key = self.w_k(k)  #(1,seq_len,d_model)
        value = self.w_v(v) #(1,seq_len,d_model)

        #(1,seq_len,d_model) -> (1,seq_len,h,d_k) -> (1,h,seq_len,d_k)
        #we didvide the last dim d_model into h,dk and then we do transpose
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        #we call our own function attention
        x,self.attention_scores =MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        #(1,h,seq_len,d_k) -> (1,seq_len,h,d_k) -> (1,seq_len,d_model)
        #we again transpose and then change the last two dim to one dim by multiplying and second dim is -1 means the pytorch calculate on its own as the dimension multilpcation remain same . it is the safe side to avaoid confusion
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)

        # (1,seq_len,d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module): #the shortcut layer or skipped connection
    def __init__(self,features:int,dropout:float ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features) # layer normalization for residual connection

    def forward(self,x,sublayer):

        #apply the residual connection with layer normalization and dropout
        #sublayer is the callable function like ffn or mha or we can call prev layer
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):#the big encoder block that contain mha fnn and two dropout in one block 

    def __init__(self,features:int,self_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout:float ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block  = feed_forward_block
        #we use module list instead of plain list so our model knows to save and train this params or if its normal list it will skip
        self.residual_connections = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(2)])

    def forward(self,x,src_mask):

        #residual connection between x and the output of the self attention block 
        #we use lambda as we are only passing one argument in residual connection in froward pass but we want 4 argument in self attention block 
        x =self.residual_connections[0](x,lambda x : self.self_attention_block(x,x,x,src_mask))
        #residual connection between x and the output of the feed forward block
        #here we have only one argument so that why no lambda
        x = self.residual_connections[1](x,self.feed_forward_block)

        return x


class Encoder(nn.Module): # as we have total n no of encoders here n is layers length

    def __init__(self,features:int,layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features) # layer normalization for encoder output

    def forward(self,x,src_mask):
        #apply the encoder blocks
        for layer in self.layers:
            x = layer(x,src_mask)

        #apply the layer normalization to the encoder output as we are using prenorm so to normalize the x and everything again we use norm again
        # it is outside of loop as we dont normalize the original x we keep it as same and only at last we apply norm to it 
        return self.norm(x)



class DecoderBlock(nn.Module):#class for one decoder block
    def __init__(self,features:int,self_attention_block:MultiHeadAttentionBlock,encoder_decoder_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.encoder_decoder_attention_block = encoder_decoder_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(3)])

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        #src mask is the mask for encoder we use in training and tgt mask is the mask of the output
        #residual connection between x and the output of the self attention block 
        x =self.residual_connections[0](x,lambda x : self.self_attention_block(x,x,x,tgt_mask))
        #residual connection between x and the output of the encoder decoder attention block
        #here q is the one come from decoder and k,v are coming from encoder suggested by paper
        x =self.residual_connections[1](x,lambda x : self.encoder_decoder_attention_block(x,encoder_output,encoder_output,src_mask))
        #residual connection between x and the output of the feed forward block
        x = self.residual_connections[2](x,self.feed_forward_block)

        return x


class Decoder(nn.Module):# as we have total n no of decoders here n is layers length

    def __init__(self,features:int,layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features) # layer normalization for decoder output

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        #apply the decoder blocks
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)

        #apply the layer normalization to the decoder output
        return self.norm(x)


class ProjectionLayer(nn.Module):# the last linear layer thta convert dmodel to vocab size

    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        #(batch,seq_len,d_model)->(batch,seq_len,vocab_size)
        return self.proj(x)



class Transformer(nn.Module):#this is the final transformer block which include everythinh 

    def __init__(self,encoder:Encoder,decoder:Decoder,source_embedding:InputEmbedding,
                 target_embedding:InputEmbedding,source_positional_embedding:PositionalEmbedding,target_positional_embedding:PositionalEmbedding,projection_layer:ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.source_positional_embedding = source_positional_embedding
        self.target_positional_embedding = target_positional_embedding
        self.projection_layer = projection_layer

    def encode(self,src,src_mask):#everything related to encoder it runs everytime with new batch

        #apply the source embedding and positional embedding to the source sequence
        src = self.source_embedding(src) #we apply input embedding
        src = self.source_positional_embedding(src)#we apply position embedding

        #apply the encoder to the source sequence
        return self.encoder(src,src_mask) #we apply the encoder we have n blocks of Encoderblock 

    def decode(self,encoder_output,src_mask,tgt,tgt_mask):#everything related to decoder it runs everytime with new batch
        #apply the target embedding and positional embedding to the target sequence
        tgt = self.target_embedding(tgt) #we apply input embedding
        tgt = self.target_positional_embedding(tgt)#we apply position embedding

        #apply the decoder to the target sequence
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)#we apply the decoder we have n blocks of Decoderblock

    def project(self,x):#the final projection layer it runs everytime with new batch

        #apply the projection layer to the decoder output
        return self.projection_layer(x)


#here we are calling the transforemre with real values
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size) # input embedding initialization
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size) # output embedding initialization

    # Create the positional encoding layers
    src_pos = PositionalEmbedding(d_model, src_seq_len, dropout) #position initialization
    tgt_pos = PositionalEmbedding(d_model, tgt_seq_len, dropout) #position initialization
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block) #here we append all

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block) #here we append all
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks)) #here we initialize the Encoder
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks)) #here we initialize the Decoder
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size) #here initialize the projection layer
    
    # call the transformer with all the arguments 
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters so we dont start with random unpredicatble weights . they would be good random
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
