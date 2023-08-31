# ECG影像處理
> 此專案含有ECG影像處理傳統方法、深度學習方法、影像處理評估方法、模型可視化，提供ECG影像不同的前處理方式作使用。
> 
## 傳統的影像前處理方式
> 這包程式碼中含有ECG影像裁剪方法以及三種前處理方式，提供ECG影像不同的前處理方式作使用，需要先對ECG影像裁剪再作前處理。
```python
def extract_box_content(folder_path, save_path)
```
> 用途 : ECG影像剪裁，去除不必要文字部分。
> folder_path 為ECG原始影像路徑，save_path為儲存之目標路徑。
```python
folder_path = '/content/drive/MyDrive/Colab Notebooks/ECG前處理相關方法實作/data/ECG訊號轉影像資料集/ECG_process/ecg_plot(grid)'
save_path = '/content/drive/MyDrive/Colab Notebooks/ECG前處理相關方法實作/data/ECG訊號轉影像資料集/ECG_process/ecg_plot(grid)'
extract_box_content(folder_path, save_path)
```
>
```python
def img_resize(image, target_height)
```
> 用途 : 調整ECG影像大小方便模型輸入與資料儲存。
> image 為輸入之ECG影像，target_height為目標高度。
>
```python
def process_image_filter(folder_path, save_path)
```
> 用途 : 提高ECG影像品質、去除雜訊、二值化，適用於影像中背景雜訊與曲線深淺差異大，會些微失真。
> folder_path 為ECG原始影像路徑，save_path為儲存之目標路徑。
```python
folder_path = '/content/drive/MyDrive/Colab Notebooks/ECG前處理相關方法實作/data/ECG訊號轉影像資料集/ECG_process/ecg_plot(grid)'
save_path = '/content/drive/MyDrive/Colab Notebooks/ECG前處理相關方法實作/data/ECG訊號轉影像資料集/ECG_process/ECG_process_filter'
process_image_filter(folder_path, save_path)
```
>
```python
def process_image_inrange(folder_path, save_path, color)
```
> 用途 : 用顏色取出ECG影像波型，適用於影像中背景雜訊與曲線顏色差異大，較不易失真。
> folder_path 為ECG原始影像路徑，save_path為儲存之目標路徑，color為目標波型顏色，共可取出藍色、黑色、灰色、白色、紅色、橘色、黃色、綠色、紫色九種色彩。
```python
folder_path = '/content/drive/MyDrive/Colab Notebooks/ECG前處理相關方法實作/data/ECG訊號轉影像資料集/ECG_process/ecg_plot(grid)'
save_path = '/content/drive/MyDrive/Colab Notebooks/ECG前處理相關方法實作/data/ECG訊號轉影像資料集/ECG_process/ECG_process_inrange'
process_image_inrange(folder_path, save_path, color="blue")
```
>
```python
def process_image_connected(folder_path, save_path)
```
> 用途 : 用連通區域取出ECG影像波型，適用於影像中背景雜訊與曲線深淺差異大，較不易失真。
> folder_path 為ECG原始影像路徑，save_path為儲存之目標路徑。
```python
folder_path = '/content/drive/MyDrive/Colab Notebooks/ECG前處理相關方法實作/data/ECG訊號轉影像資料集/ECG_process/ecg_plot(grid)'
save_path = '/content/drive/MyDrive/Colab Notebooks/ECG前處理相關方法實作/data/ECG訊號轉影像資料集/ECG_process/ECG_process_connected'
process_image_connected(folder_path, save_path)
```
> 
## U-Net
> 這包程式碼中含有U-Net影像切割方法(訓練及測試)，提供ECG影像深度學習的前處理方式作使用，取出影像ROI區域。
* 使用方法
  ```python
  def unet_training(img_w, img_h, train_path, image_folder, mask_folder, model_save_path, batch_size=1)
  ```
  >使用unet_training訓練U-Net模型，其中img_w、img_h分別表示輸入影像之寬、高，image_folder為ECG訓練影像資料夾名稱，mask_folder為訓練遮罩資料夾名稱，model_save_path為模型權重儲存目標路徑，batch_size可依照電腦效能做設定。
  ```python
  unet_training(2007, 727, "/content/drive/MyDrive/Colab Notebooks/ECG前處理相關方法實作/data/ECG訊號轉影像資料集/ECG_process/new_train_data","image", "label", "/content/drive/MyDrive/Colab Notebooks/ECG前處理相關方法實作/data/ECG訊號轉影像資料集/unet_membrane.hdf5")
  ```
  >
   ```python
  def unet_testing(img_w, img_h, num_image, test_path, save_path, model_weight_path)
  ```
  >使用unet_testing載入模型權重測試U-Net模型，並產出切割結果，其中img_w、img_h分別表示輸入影像之寬、高，num_image表示測試的影像張數，test_path為ECG測試影像路徑，save_path為分割結果儲存路徑，model_weight_path為載入模型權重儲存路徑。
  ```python
  unet_testing(2007, 727, 250, "/content/drive/MyDrive/Colab Notebooks/ECG前處理相關方法實作/data/ECG訊號轉影像資料集/ECG_process/new_test_data/image",
             "/content/drive/MyDrive/Colab Notebooks/ECG前處理相關方法實作/data/ECG訊號轉影像資料集/ECG_process/ECG_process_unet",
             "/content/drive/MyDrive/Colab Notebooks/ECG前處理相關方法實作/data/ECG訊號轉影像資料集/unet_membrane.hdf5")
  ```
>
## ECG影像標準
> 這包程式碼中含有四種ECG影像處理方法的評估指標，根據影像雜訊去除的程度提供ECG影像客觀的評估結果。
* 使用方法
  ```python
  def method_evaluation(method_image_path,ground_truth_path)
  ```
  > 使用method_evaluation計算處理後的影像與無雜訊影像兩者的相似程度，依照雜訊去除的程度判斷前處理的優劣，總共會使用MSE、PSNR、SSIM、Dice係數四種指標，其中method_image_path為前處理結果資料路徑，ground_truth_path為無雜訊影像路徑。
  ```python
  method_image_path = "/content/drive/MyDrive/Colab Notebooks/ECG前處理相關方法實作/data/ECG訊號轉影像資料集/ECG_process/ECG_process_filter"
  ground_truth_path = "/content/drive/MyDrive/Colab Notebooks/ECG前處理相關方法實作/data/ECG訊號轉影像資料集/ECG_process/label"
  avgpsnr, avgmse, avgssim, avgdice = method_evaluation(method_image_path,ground_truth_path)
  print("PSNR:", avgpsnr)
  print("MSE:", avgmse)
  print("SSIM:", avgssim)
  print("DICE:", avgdice)
  ```
>
## 模型可視化
> 這包程式碼中含有以Grad CAM進行模型可視化的方式，能夠突顯模型關注區域來解釋模型。
* 使用方法
  ```python
  def grad_cam(cnn_model_output, classifier_model, disease_name, folder_path, org_folder_path, save_path)
  ```
  > 使用grad_cam生成模型可視化的影像，依照最後一層CNN的梯度以及模型的分類結果得出分布圖，並轉換為Heatmap，其中cnn_model_output為CNN模型的最後一層的輸出張量，classifier_model為分類模型(FCN)的部分，disease_name表示類別(不同疾病的名稱列表)，folder_path為模型測試影像的資料夾路徑(模型測試影像大小已被調整過)，org_folder_path為模型測試影像的原始影像(未前處理及調整大小，僅裁剪之影像)，save_path為儲存之目標路徑。
  ```python
  # 需準備CNN網路的最後一層輸出以及分類網路
  # 載入保存好的模型
  loaded_model = tf.keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/ECG前處理相關方法實作/paper2_COVID19_filter_0808/paper2_COVID19_filter_0808")
  # 輸出模型架構
  print(loaded_model.summary())
  # 依照模型架構取出最後一層CNN以及分類模型
  # 將CNN模型取出
  model = loaded_model.get_layer("model")
  
  # 取得CNN模型輸出大小
  classifier_input = tf.keras.Input(shape=model.output.shape[1:])
  x = classifier_input
  
  # 取得分類模型
  for layer_name in ["global_average_pooling2d", "dense"]:
    x = loaded_model.get_layer(layer_name)(x)
  classifier_model = tf.keras.Model(classifier_input, x)
  print(classifier_model.summary())
  
  # 標籤名稱
  disease_name = ['COVID19', 'HB', 'MI', 'Normal', 'PMI']
  
  folder_path = "/content/drive/MyDrive/Colab Notebooks/ECG前處理相關方法實作/data/Grad_CAM/image"
  org_folder_path = "/content/drive/MyDrive/Colab Notebooks/ECG前處理相關方法實作/data/Grad_CAM/org_image"
  save_path = "/content/drive/MyDrive/Colab Notebooks/ECG前處理相關方法實作/data/Grad_CAM/save"
  grad_cam(model, classifier_model, disease_name, folder_path, org_folder_path, save_path)
  ```
