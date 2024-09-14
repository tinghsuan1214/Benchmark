import os

def replace_flops_with_macs(directory):
    # 遍歷資料夾和子資料夾
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                
                # 讀取檔案內容
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # 找到並替換 "FLOPs" 為 "MACs"
                if "Model FLOPs:" in content:
                    updated_content = content.replace("Model FLOPs:", "Model MACs:")
                    
                    # 將替換後的內容寫回檔案
                    with open(file_path, 'w') as f:
                        f.write(updated_content)
                    print(f"Updated: {file_path}")

# 指定根目錄路徑
root_directory = 'C:/Users/tingh/OneDrive/桌面/git/Benchmark/Benchmark/Search_Space_1/Nodes_3/Linear_2/Linaer_2_1'
replace_flops_with_macs(root_directory)
