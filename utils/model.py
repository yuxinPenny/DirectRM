import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class denovoModel(nn.Module):
    def __init__(self):
        super(denovoModel, self).__init__()
        self.lstm = nn.LSTM(input_size= 16,hidden_size=16,bidirectional=True)
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(288, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,2),
            nn.Softmax(dim=1)
        )

    def forward(self,seq,stat,bse):
        x = torch.cat((seq.float(),stat.float(),bse.float()),dim=2)
        # x = torch.cat((seq.float(),stat.float()),dim=2)
        h, (_,_) = self.lstm(x)
        h = h.reshape(h.shape[0],-1)
        Y = self.classifier(h)
        return Y

class Attention(nn.Module):
    def __init__(self,input_dim):
        super(Attention,self).__init__()
        self.W = nn.Linear(input_dim,input_dim)
        self.V = nn.Linear(input_dim,1)
    def forward(self,x):
        x = self.W(x)
        x = torch.tanh(x)
        A = self.V(x)
        A = nn.Softmax(dim=1)(A)
        return A

class PositionalEncoding(nn.Module):
    def __init__(self,seq_len,input_dim):
        super(PositionalEncoding,self).__init__()
        pe = torch.zeros(seq_len,input_dim)
        pos = torch.arange(0,seq_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0,input_dim,2,dtype=torch.float) * (-math.log(1) / input_dim)
                              ))
        pe[:,0::2] = torch.sin(pos.float() * div_term)
        pe[:,1::2] = torch.cos(pos.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self,x):
        return x + self.pe[:,:x.size(1)]

# v1 attention only
class subModel1(nn.Module):
    def __init__(self,dim1=16,dim2=160):
        super(subModel1,self).__init__()
        # dim = 320
        self.attention = Attention(dim1)
        self.fc = nn.Sequential(
            nn.Linear(dim2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,2),
            nn.Softmax(dim=1)
        )
    def forward(self,seq,stat,bse):
        x1 = torch.cat((seq.float(),stat.float(),bse.float()),dim=2)
        A = self.attention(x1)
        A = torch.transpose(A,1,2)
        M = torch.bmm(A,x1)
        M = M.reshape(M.size()[0],-1)

        x1 = x1.reshape(x1.size()[0],-1)
        x = torch.cat([M,x1],axis = 1)
        y = self.fc(x)
        y = y[:,1].reshape(-1,1)
        return y,A

# v2: attention + bidirectional LSTM
class subModel2(nn.Module):
    def __init__(self,dim1=16,dim2=640):
        super(subModel2,self).__init__()
        # dim = 320
        self.attention = Attention(64)
        self.lstm = nn.LSTM(input_size= dim1,hidden_size=32,bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(dim2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,2),
            nn.Softmax(dim=1)
        )
    def forward(self,seq,stat,bse):
        x1 = torch.cat((seq.float(),stat.float(),bse.float()),dim=2)

        H,(_,_) = self.lstm(x1)
        A = self.attention(H)
        A = torch.transpose(A,1,2)
        M = torch.bmm(A,H)
        M = M.reshape(M.size()[0],-1)

        H = H.reshape(H.size()[0],-1)
        x = torch.cat([M,H],axis = 1)
        y = self.fc(x)
        y = y[:,1].reshape(-1,1)
        return y,A

# v3: attention + positional encoding
class subModel3(nn.Module):
    def __init__(self,dim1=16,dim2=160):
        super(subModel3,self).__init__()
        # dim = 320
        self.attention = Attention(dim1)
        self.pe = PositionalEncoding(seq_len=9,input_dim=dim1)
        self.event_lstm = nn.LSTM(input_size= 80,hidden_size=64,bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(dim2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,2),
            nn.Softmax(dim=1)
        )
    def forward(self,seq,stat,bse):
        x1 = torch.cat((seq.float(),stat.float(),bse.float()),dim=2)
        x1 = self.pe(x1)
        A = self.attention(x1)
        A = torch.transpose(A,1,2)
        M = torch.bmm(A,x1)
        M = M.reshape(M.size()[0],-1)

        x1 = x1.reshape(x1.size()[0],-1)
        x = torch.cat([M,x1],axis = 1)
        y = self.fc(x)
        y = y[:,1].reshape(-1,1)
        return y,A

# v3: attention + positional encoding + bidirectional LSTM
class subModel4(nn.Module):
    def __init__(self,dim1=16,dim2=640):
        super(subModel4,self).__init__()
        # dim = 320
        self.attention = Attention(64)
        self.pe = PositionalEncoding(seq_len=9,input_dim=dim1)
        self.lstm = nn.LSTM(input_size= dim1,hidden_size=32,bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(dim2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,2),
            nn.Softmax(dim=1)
        )
    def forward(self,seq,stat,bse):

        x1 = torch.cat((seq.float(),stat.float(),bse.float()),dim=2)
        x1 = self.pe(x1)
        H,(_,_) = self.lstm(x1)
        A = self.attention(H)
        A = torch.transpose(A,1,2)
        M = torch.bmm(A,H)
        M = M.reshape(M.size()[0],-1)

        H = H.reshape(H.size()[0],-1)
        x = torch.cat([M,H],axis = 1)
        y = self.fc(x)
        y = y[:,1].reshape(-1,1)
        return y,A

class FeatureExtractor1(nn.Module):
    def __init__(self):
        super(FeatureExtractor1,self).__init__()
        self.lstm = nn.LSTM(input_size= 16,hidden_size=8,bidirectional=True)
    def forward(self,seq,stat,bse):
        x = torch.cat((seq.float(),stat.float(),bse.float()),dim=2)
        H, (_,_) = self.lstm(x)
        H = H.reshape(H.size()[0],-1)
        return H

class FeatureExtractor2(nn.Module):
    def __init__(self):
        super(FeatureExtractor2,self).__init__()
        self.lstm = nn.LSTM(input_size= 16,hidden_size=8,bidirectional=True)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=3, out_channels=2, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
    def forward(self,seq,stat,bse):
        x = torch.cat((seq.float(),stat.float(),bse.float()),dim=2)
        H, (_,_) = self.lstm(x)
        H = H.reshape(H.size()[0],1,-1)
        x2 = self.cnn(H)
        x2 = x2.reshape(x2.size()[0],-1)
        return x2

class FeatureExtractor3(nn.Module):
    def __init__(self):
        super(FeatureExtractor3,self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=3, out_channels=2, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
    def forward(self,seq,stat,bse):
        x = torch.cat((seq.float(),stat.float(),bse.float()),dim=2)
        x = x.reshape(x.size()[0],1,-1)
        x2 = self.cnn(x)
        x2 = x2.reshape(x2.size()[0],-1)
        return x2
# attention + feature extractor 1
class subModel5(nn.Module):
    def __init__(self,dim1=16,dim2=160):
        super(subModel5,self).__init__()
        self.attention = Attention(dim1)
        self.feature_extractor = FeatureExtractor1()
        self.fc = nn.Sequential(
            nn.Linear(dim2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,2),
            nn.Softmax(dim=1)
        )
    def forward(self,seq,stat,bse):
        x1 = torch.cat((seq.float(),stat.float(),bse.float()),dim=2)
        A = self.attention(x1)
        A = torch.transpose(A,1,2)
        M = torch.bmm(A,x1)
        M = M.reshape(M.size()[0],-1)

        x2 = self.feature_extractor(seq,stat,bse)
        x = torch.cat([M,x2],axis = 1)
        y = self.fc(x)
        y = y[:,1].reshape(-1,1)
        return y,A

# attention + feature extractor 2
class subModel6(nn.Module):
    def __init__(self,dim1=16,dim2=86):
        super(subModel6,self).__init__()
        self.attention = Attention(dim1)
        self.feature_extractor = FeatureExtractor2()
        self.fc = nn.Sequential(
            nn.Linear(dim2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,2),
            nn.Softmax(dim=1)
        )
    def forward(self,seq,stat,bse):
        x1 = torch.cat((seq.float(),stat.float(),bse.float()),dim=2)
        A = self.attention(x1)
        A = torch.transpose(A,1,2)
        M = torch.bmm(A,x1)
        M = M.reshape(M.size()[0],-1)

        x2 = self.feature_extractor(seq,stat,bse)
        x = torch.cat([M,x2],axis = 1)
        y = self.fc(x)
        y = y[:,1].reshape(-1,1)
        return y,A

# attention + feature extractor 3
class subModel7(nn.Module):
    def __init__(self,dim1=16,dim2=86):
        super(subModel7,self).__init__()
        self.attention = Attention(dim1)
        self.feature_extractor = FeatureExtractor3()
        self.fc = nn.Sequential(
            nn.Linear(dim2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,2),
            nn.Softmax(dim=1)
        )
    def forward(self,seq,stat,bse):
        x1 = torch.cat((seq.float(),stat.float(),bse.float()),dim=2)
        A = self.attention(x1)
        A = torch.transpose(A,1,2)
        M = torch.bmm(A,x1)
        M = M.reshape(M.size()[0],-1)

        x2 = self.feature_extractor(seq,stat,bse)
        x = torch.cat([M,x2],axis = 1)
        y = self.fc(x)
        y = y[:,1].reshape(-1,1)
        return y,A

class subModel8(nn.Module):
    def __init__(self,dim1=16,dim2=160):
        super(subModel8,self).__init__()
        self.attention = Attention(dim1)
        self.feature_extractor = FeatureExtractor1()
        self.fc = nn.Sequential(
            nn.Linear(dim2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,2),
            nn.Softmax(dim=1)
        )
    def forward(self,seq,stat,bse,x2):
        x1 = torch.cat((seq.float(),stat.float(),bse.float()),dim=2)
        A = self.attention(x1)
        A = torch.transpose(A,1,2)
        M = torch.bmm(A,x1)
        M = M.reshape(M.size()[0],-1)

        x = torch.cat([M,x2],axis = 1)
        y = self.fc(x)
        y = y[:,1].reshape(-1,1)
        return y,A

class mlModel1(nn.Module):
    def __init__(self):
        super(mlModel1, self).__init__()
        self.ac4c = subModel1()
        self.m1a = subModel1()
        self.m5c = subModel1()
        self.m6a = subModel1()
        self.m7g = subModel1()
        self.psi = subModel1()

    def forward(self, seq,stat,bse):
        ac4c_y, ac4c_A = self.ac4c(seq,stat,bse)
        m1a_y, m1a_A = self.m1a(seq,stat,bse)
        m5c_y, m5c_A = self.m5c(seq,stat,bse)
        m6a_y, m6a_A = self.m6a(seq,stat,bse)
        m7g_y, m7g_A = self.m7g(seq,stat,bse)
        psi_y, psi_A = self.psi(seq,stat,bse)
        Y = torch.cat([ac4c_y,m1a_y,m5c_y,m6a_y,m7g_y,psi_y],dim=1)
        A = torch.cat([ac4c_A,m1a_A,m5c_A,m6a_A,m7g_A,psi_A],dim=1)
        return Y,A

class mlModel2(nn.Module):
    def __init__(self):
        super(mlModel2, self).__init__()
        self.ac4c = subModel2()
        self.m1a = subModel2()
        self.m5c = subModel2()
        self.m6a = subModel2()
        self.m7g = subModel2()
        self.psi = subModel2()

    def forward(self, seq,stat,bse):
        ac4c_y, ac4c_A = self.ac4c(seq,stat,bse)
        m1a_y, m1a_A = self.m1a(seq,stat,bse)
        m5c_y, m5c_A = self.m5c(seq,stat,bse)
        m6a_y, m6a_A = self.m6a(seq,stat,bse)
        m7g_y, m7g_A = self.m7g(seq,stat,bse)
        psi_y, psi_A = self.psi(seq,stat,bse)
        Y = torch.cat([ac4c_y,m1a_y,m5c_y,m6a_y,m7g_y,psi_y],dim=1)
        A = torch.cat([ac4c_A,m1a_A,m5c_A,m6a_A,m7g_A,psi_A],dim=1)
        return Y,A

class mlModel3(nn.Module):
    def __init__(self):
        super(mlModel3, self).__init__()
        self.ac4c = subModel3()
        self.m1a = subModel3()
        self.m5c = subModel3()
        self.m6a = subModel3()
        self.m7g = subModel3()
        self.psi = subModel3()

    def forward(self, seq,stat,bse):
        ac4c_y, ac4c_A = self.ac4c(seq,stat,bse)
        m1a_y, m1a_A = self.m1a(seq,stat,bse)
        m5c_y, m5c_A = self.m5c(seq,stat,bse)
        m6a_y, m6a_A = self.m6a(seq,stat,bse)
        m7g_y, m7g_A = self.m7g(seq,stat,bse)
        psi_y, psi_A = self.psi(seq,stat,bse)
        Y = torch.cat([ac4c_y,m1a_y,m5c_y,m6a_y,m7g_y,psi_y],dim=1)
        A = torch.cat([ac4c_A,m1a_A,m5c_A,m6a_A,m7g_A,psi_A],dim=1)
        return Y,A

class mlModel4(nn.Module):
    def __init__(self):
        super(mlModel4, self).__init__()
        self.ac4c = subModel4()
        self.m1a = subModel4()
        self.m5c = subModel4()
        self.m6a = subModel4()
        self.m7g = subModel4()
        self.psi = subModel4()

    def forward(self, seq,stat,bse):
        ac4c_y, ac4c_A = self.ac4c(seq,stat,bse)
        m1a_y, m1a_A = self.m1a(seq,stat,bse)
        m5c_y, m5c_A = self.m5c(seq,stat,bse)
        m6a_y, m6a_A = self.m6a(seq,stat,bse)
        m7g_y, m7g_A = self.m7g(seq,stat,bse)
        psi_y, psi_A = self.psi(seq,stat,bse)
        Y = torch.cat([ac4c_y,m1a_y,m5c_y,m6a_y,m7g_y,psi_y],dim=1)
        A = torch.cat([ac4c_A,m1a_A,m5c_A,m6a_A,m7g_A,psi_A],dim=1)
        return Y,A


class mlModel5(nn.Module):
    def __init__(self):
        super(mlModel5, self).__init__()
        self.ac4c = subModel5()
        self.m1a = subModel5()
        self.m5c = subModel5()
        self.m6a = subModel5()
        self.m7g = subModel5()
        self.psi = subModel5()

    def forward(self, seq,stat,bse):
        ac4c_y, ac4c_A = self.ac4c(seq,stat,bse)
        m1a_y, m1a_A = self.m1a(seq,stat,bse)
        m5c_y, m5c_A = self.m5c(seq,stat,bse)
        m6a_y, m6a_A = self.m6a(seq,stat,bse)
        m7g_y, m7g_A = self.m7g(seq,stat,bse)
        psi_y, psi_A = self.psi(seq,stat,bse)
        Y = torch.cat([ac4c_y,m1a_y,m5c_y,m6a_y,m7g_y,psi_y],dim=1)
        A = torch.cat([ac4c_A,m1a_A,m5c_A,m6a_A,m7g_A,psi_A],dim=1)
        return Y,A

class mlModel6(nn.Module):
    def __init__(self):
        super(mlModel6, self).__init__()
        self.ac4c = subModel6()
        self.m1a = subModel6()
        self.m5c = subModel6()
        self.m6a = subModel6()
        self.m7g = subModel6()
        self.psi = subModel6()

    def forward(self, seq,stat,bse):
        ac4c_y, ac4c_A = self.ac4c(seq,stat,bse)
        m1a_y, m1a_A = self.m1a(seq,stat,bse)
        m5c_y, m5c_A = self.m5c(seq,stat,bse)
        m6a_y, m6a_A = self.m6a(seq,stat,bse)
        m7g_y, m7g_A = self.m7g(seq,stat,bse)
        psi_y, psi_A = self.psi(seq,stat,bse)
        Y = torch.cat([ac4c_y,m1a_y,m5c_y,m6a_y,m7g_y,psi_y],dim=1)
        A = torch.cat([ac4c_A,m1a_A,m5c_A,m6a_A,m7g_A,psi_A],dim=1)
        return Y,A

class mlModel7(nn.Module):
    def __init__(self):
        super(mlModel7, self).__init__()
        self.ac4c = subModel7()
        self.m1a = subModel7()
        self.m5c = subModel7()
        self.m6a = subModel7()
        self.m7g = subModel7()
        self.psi = subModel7()

    def forward(self, seq,stat,bse):
        ac4c_y, ac4c_A = self.ac4c(seq,stat,bse)
        m1a_y, m1a_A = self.m1a(seq,stat,bse)
        m5c_y, m5c_A = self.m5c(seq,stat,bse)
        m6a_y, m6a_A = self.m6a(seq,stat,bse)
        m7g_y, m7g_A = self.m7g(seq,stat,bse)
        psi_y, psi_A = self.psi(seq,stat,bse)
        Y = torch.cat([ac4c_y,m1a_y,m5c_y,m6a_y,m7g_y,psi_y],dim=1)
        A = torch.cat([ac4c_A,m1a_A,m5c_A,m6a_A,m7g_A,psi_A],dim=1)
        return Y,A

class mlModel8(nn.Module):
    def __init__(self):
        super(mlModel8, self).__init__()
        self.feature_extractor = FeatureExtractor1()
        self.ac4c = subModel8()
        self.m1a = subModel8()
        self.m5c = subModel8()
        self.m6a = subModel8()
        self.m7g = subModel8()
        self.psi = subModel8()

    def forward(self, seq,stat,bse):
        x2 = self.feature_extractor(seq,stat,bse)
        ac4c_y, ac4c_A = self.ac4c(seq,stat,bse,x2)
        m1a_y, m1a_A = self.m1a(seq,stat,bse,x2)
        m5c_y, m5c_A = self.m5c(seq,stat,bse,x2)
        m6a_y, m6a_A = self.m6a(seq,stat,bse,x2)
        m7g_y, m7g_A = self.m7g(seq,stat,bse,x2)
        psi_y, psi_A = self.psi(seq,stat,bse,x2)
        Y = torch.cat([ac4c_y,m1a_y,m5c_y,m6a_y,m7g_y,psi_y],dim=1)
        A = torch.cat([ac4c_A,m1a_A,m5c_A,m6a_A,m7g_A,psi_A],dim=1)
        return Y,A