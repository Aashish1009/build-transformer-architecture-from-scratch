# importing the requirements
import torch
import torch.nn as nn
import math

#class for the input embedding
class InputEmbedding(nn.Module):

    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)#parameter order matter

    def forward(self,x):
        #in the paper they multiply the embedding by sqrt(d_model) to scale the embedding
        return self.embedding(x)* math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model:int, seq_len:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #create the positional embedding (seq_len,d_model)
        pe = torch.zeros(seq_len,d_model)

        #create the positional encoding (seq_len,1)
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)

        #create the div term to scale the positional encoding
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))

        #apply the positional encoding to the positional embedding
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)

        #add a batch dimension to the positional embedding (1,seq_len,d_model)
        pe = pe.unsqueeze(0)

        #register the buffer of the positional embedding
        self.register_buffer("pe",pe)

    def forward(self,x):

        #add the positional encoding to the sequence of tokens (1,seq_len,d_model)
        x= x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)

        #apply the dropout
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self,eps:float = 1e-6 ):
        super().__init__()
        self.eps = eps # small value to avoid division by zero
        self.alpha = nn.Parameter(torch.ones(1)) #scale parameter learned during training 
        self.bias = nn.Parameter(torch.zeros(1)) #bias parameter learned during training

    def forward(self,x):

        #calculate the mean and standard deviation
        mean = x.mean(dim=-1,keepdim=True) #keepdim=True to maintain the same shape
        std = x.std(dim=-1,keepdim=True)  #keepdim=True to maintain the same shape

        #apply the layer normalization
        return self.alpha *(x-mean) / (std +self.eps)  + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self,d_model:int,d_ff:int,dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)#(d_model,d_ff) expansion layer
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff,d_model)#(d_ff,d_model) compression layer

    def forward(self,x):

        #apply the feed forward block with relu activation and dropout
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model:int,h:int,dropout:float ):
        super().__init__()
        self.d_model = d_model #dimension of the model
        self.h = h #number of heads
        assert d_model %h==0,"dmodel is not divisible by h"

        self.dropout = nn.Dropout(dropout)
        self.d_k = d_model//h #dimension of dk

        #define the linear layers for query, key, value and output
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)

        self.w_o = nn.Linear(d_model,d_model)


    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k = query.shape[-1] # dimension of dk

        attention_scores = query @ key.transpose(-2,-1) / math.sqrt(d_k) #(1,h,seq_len,seq_len)

        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)

        attention_scores = attention_scores.softmax(dim=-1) #(1,h,seq_len,seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value) , attention_scores


    def forward(self,k,q,v,mask):

        query = self.w_q(q) #(1,seq_len,d_model)
        key = self.w_k(k)  #(1,seq_len,d_model)
        value = self.w_v(v) #(1,seq_len,d_model)

        #(1,seq_len,d_model) -> (1,seq_len,h,d_k) -> (1,h,seq_len,d_k)
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x,self.attention_scores =MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        #(1,h,seq_len,d_k) -> (1,seq_len,h,d_k) -> (1,seq_len,d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)

        # (1,seq_len,d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self,dropout:float ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization() # layer normalization for residual connection

    def forward(self,x,sublayer):

        #apply the residual connection with layer normalization and dropout
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self,self_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout:float ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block  = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self,x,src_mask):

        #residual connection between x and the output of the self attention block 
        x =self.residual_connections[0](x,lambda x : self.self_attention_block(x,x,x,src_mask))
        #residual connection between x and the output of the feed forward block
        x = self.residual_connections[1](x,self.feed_forward_block)

        return x


class Encoder(nn.Module):

    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization() # layer normalization for encoder output

    def forward(self,x,src_mask):
        #apply the encoder blocks
        for layer in self.layers:
            x = layer(x,src_mask)

        #apply the layer normalization to the encoder output
        return self.norm(x)



class DecoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttentionBlock,encoder_decoder_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.encoder_decoder_attention_block = encoder_decoder_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self,x,encoder_output,src_mask,tgt_mask):

        #residual connection between x and the output of the self attention block 
        x =self.residual_connections[0](x,lambda x : self.self_attention_block(x,x,x,tgt_mask))
        #residual connection between x and the output of the encoder decoder attention block
        x =self.residual_connections[1](x,lambda x : self.encoder_decoder_attention_block(x,encoder_output,encoder_output,src_mask))
        #residual connection between x and the output of the feed forward block
        x = self.residual_connections[2](x,self.feed_forward_block)

        return x


class Decoder(nn.Module):

    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization() # layer normalization for decoder output

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        #apply the decoder blocks
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)

        #apply the layer normalization to the decoder output
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        #apply the projection layer to the decoder output and return the log softmax of the output
        return torch.log_softmax(self.proj(x),dim=-1)



class Transformer(nn.Module):

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

    def encode(self,src,src_mask):

        #apply the source embedding and positional embedding to the source sequence
        src = self.source_embedding(src)
        src = self.source_positional_embedding(src)

        #apply the encoder to the source sequence
        return self.encoder(src,src_mask)

    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        #apply the target embedding and positional embedding to the target sequence
        tgt = self.target_embedding(tgt)
        tgt = self.target_positional_embedding(tgt)

        #apply the decoder to the target sequence
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)

    def project(self,x):

        #apply the projection layer to the decoder output
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEmbedding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEmbedding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
