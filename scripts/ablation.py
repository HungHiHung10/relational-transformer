from rt.main import main
from rt.tasks import all_tasks, forecast_tasks

# --- BƯỚC 1: TẠO HÀM CẮT BỎ (ABLATION FUNCTION) ---
def apply_self_label_ablation(task_list):
    """
    Hàm này duyệt qua danh sách task và đưa target_column (phần tử thứ 3)
    vào danh sách leakage_columns (phần tử thứ 4) để che giấu nhãn quá khứ.
    """
    ablated_tasks = []
    for db_name, table_name, target_column, leakage_columns in task_list:
        # Tạo một bản sao của danh sách cột rò rỉ hiện tại
        new_leakage = list(leakage_columns)
        
        # Nếu cột mục tiêu chưa nằm trong danh sách rò rỉ thì thêm vào
        if target_column not in new_leakage:
            new_leakage.append(target_column)
            
        # Thêm task đã cắt bỏ vào danh sách mới
        ablated_tasks.append((db_name, table_name, target_column, new_leakage))
    return ablated_tasks

# --- BƯỚC 2: LỌC TASK NHƯ BÌNH THƯỜNG ---
# Baseline tasks (Gốc)
base_train_tasks = [t for t in all_tasks if t[0] == "rel-event"]
base_eval_tasks = [t for t in forecast_tasks if t[0] in ["rel-trial", "rel-f1"]]

# --- BƯỚC 3: ÁP DỤNG CẮT BỎ ---
# Chuyển đổi thành Ablated tasks
ablated_train_tasks = apply_self_label_ablation(base_train_tasks)
ablated_eval_tasks = apply_self_label_ablation(base_eval_tasks)

if __name__ == "__main__":
    main(
        # misc
        project="rt_ablation_study", # Đổi tên project để dễ phân biệt trên wandb
        eval_splits=["val", "test"],
        eval_freq=1_000,
        eval_pow2=True,
        max_eval_steps=40,
        load_ckpt_path=None,
        save_ckpt_dir="ckpts/pretrain_ablated/", # Đổi thư mục lưu để không ghi đè model gốc
        compile_=True,
        seed=0,
        
        # data - SỬ DỤNG DANH SÁCH ĐÃ CẮT BỎ
        train_tasks=ablated_train_tasks,
        eval_tasks=ablated_eval_tasks,
        
        batch_size=32,
        num_workers=2,
        max_bfs_width=256,
        # optimization
        lr=1e-3,
        wd=0.1,
        lr_schedule=True,
        max_grad_norm=1.0,
        # model
        embedding_model="all-MiniLM-L12-v2",
        d_text=384,
        seq_len=1024,
        num_blocks=12,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        max_steps=None,
        epochs=1,
    )