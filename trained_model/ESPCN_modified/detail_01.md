## model description
* model = ESPCN-modified
* output layer = Sigmoid
* train dataset = dataset/DIV2K_train_LR_bicubic_X2
* criterion = nn.MSELoss()
* optimizer = optim.Adam(net.parameters(), lr=1e-4)
* epochs = 50
* batch_size = 1
* psnr_in_valid_dataset = 26.56372080644276