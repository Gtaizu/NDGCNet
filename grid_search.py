from utils import train_evaluate
import itertools
import csv

if __name__ == "__main__":
    data_path = "ncRNADrug_data_end.pth"

    # -------------------------
    # 定义要搜索的超参数范围
    # -------------------------
    batch_size_list = [128,256,512]
    lr_list = [1e-4, 0.0005, 0.001]
    conv_layers_list = [1, 2, 3]  # RNA/Drug 卷积层统一
    num_epochs_list = [200, 300, 400]  # 可搜索的训练轮数
    # hidden_dim_list = [64, 128, 256]


    cnn_channels = 64
    kernel_sizes = [3, 5, 7]

    # -------------------------
    # 生成所有参数组合
    # -------------------------
    param_combinations = list(itertools.product(batch_size_list, lr_list, num_epochs_list, conv_layers_list))

    results = []

    # -------------------------
    # 遍历每组组合
    # -------------------------
    for batch_size, lr, num_epochs, conv_layers_list in param_combinations:
        print(f"\n=== Training with batch_size={batch_size}, lr={lr}, num_epochs={num_epochs}, conv_layers_list={conv_layers_list} ===")

        mean_results, test_results = train_evaluate(
            data_path=data_path,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            cnn_channels=cnn_channels,
            kernel_sizes=kernel_sizes,
            conv_layers=conv_layers_list,
            hidden_dim=256,
            device='cuda:3',
            use_attention=True,
            use_kfold=False
        )

        # 保存参数与结果
        results.append({
            'batch_size': batch_size,
            'lr': lr,
            'conv_layers_list': conv_layers_list,
            'num_epochs': num_epochs,
            'mean_AUC': mean_results[0],
            'mean_AUPR': mean_results[1],
            'mean_ACC': mean_results[2],
            'mean_F1': mean_results[3],
            'mean_Precision': mean_results[4],
            'mean_MCC': mean_results[5],
            'mean_Brier': mean_results[6],
            'test_AUC': test_results[0],
            'test_AUPR': test_results[1],
            'test_ACC': test_results[2],
            'test_F1': test_results[3],
            'test_Precision': test_results[4],
            'test_MCC': test_results[5],
            'test_Brier': test_results[6]
        })

    # -------------------------
    # 保存所有结果到 CSV
    # -------------------------
    csv_file = "grid_search_results.csv"
    keys = results[0].keys()
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"Grid search results saved to {csv_file}")

    # -------------------------
    # 自动选出最佳组合 (按 test_AUC)
    # -------------------------
    best_result = max(results, key=lambda x: x['test_AUC'])
    print("\n=== Best Hyperparameters Based on Test AUC ===")
    print(f"batch_size={best_result['batch_size']}, lr={best_result['lr']}, conv_layers={best_result['conv_layers']}, num_epochs={best_result['num_epochs']}")
    print(f"Mean: AUC={best_result['mean_AUC']:.4f}, AUPR={best_result['mean_AUPR']:.4f}, ACC={best_result['mean_ACC']:.4f}, "
          f"F1={best_result['mean_F1']:.4f}, Precision={best_result['mean_Precision']:.4f}, MCC={best_result['mean_MCC']:.4f}, Brier={best_result['mean_Brier']:.4f}")
    print(f"Test: AUC={best_result['test_AUC']:.4f}, AUPR={best_result['test_AUPR']:.4f}, ACC={best_result['test_ACC']:.4f}, "
          f"F1={best_result['test_F1']:.4f}, Precision={best_result['test_Precision']:.4f}, MCC={best_result['test_MCC']:.4f}, Brier={best_result['test_Brier']:.4f}")

    # -------------------------
    # 保存最佳组合到 CSV
    # -------------------------
    best_csv_file = "best_grid_search_result.csv"
    with open(best_csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerow(best_result)

    print(f"Best hyperparameter result saved to {best_csv_file}")
