#date: 2022-10-12T17:13:29Z
#url: https://api.github.com/gists/21fb5ab834338ae30c19a0e30a9af923
#owner: https://api.github.com/users/shark8me

class ConvStack1d(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        cqt_feats=217
        op_divby=2
        l1_out_channels= cqt_feats //op_divby
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv1d(output_features, l1_out_channels, 3 , padding=1),
            nn.BatchNorm1d(l1_out_channels),
            nn.ReLU()
        )
        l2_out_channels= l1_out_channels // 2
        self.cnn2 = nn.Sequential(
            nn.Conv1d(l1_out_channels, l2_out_channels, 3 , padding=1),
            nn.BatchNorm1d(l2_out_channels),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),

        )
        l3_out_channels= l2_out_channels // 2
        self.cnn3 = nn.Sequential(
            nn.Conv1d(l2_out_channels, l3_out_channels, 3 , padding=1),
            nn.BatchNorm1d(l3_out_channels),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25))
        self.fc = nn.Sequential(
            nn.Linear(l2_out_channels * l3_out_channels, output_features)
        )


    def forward(self, mel):

        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = torch.squeeze(x,dim=1)
        x = self.cnn(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x
