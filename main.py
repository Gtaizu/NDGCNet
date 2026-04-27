from utils import train_evaluate

if __name__ == "__main__":
    data_path = "ncRNADrug_data_end5.pth"

    mean_results, test_results = train_evaluate(
        data_path=data_path,
        num_epochs=300,
        batch_size=512,
        lr=0.0005,
        neg_ratio=1,
        cnn_channels=64,
        kernel_sizes=[3, 5, 7],
        hidden_dim=256,
        conv_layers=2,  # RNA和Drug卷积层数
        device='cuda:0',
        channels=1,
        use_attention=True,
        use_kfold=False
    )