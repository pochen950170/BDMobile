# ECG影像處理
> 此專案含有ECG影像處理傳統方法、深度學習方法、影像處理評估方法、模型可視化，提供ECG影像不同的前處理方式作使用。
> 
## 傳統的影像前處理方式
```python
def extract_box_content(folder_path, save_path)
```
> 用途 : ECG影像剪裁，去除不必要文字部分。
> folder_path 為ECG原始影像路徑，save_path為儲存之目標路徑。
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
>
```python
def process_image_inrange(folder_path, save_path, color)
```
> 用途 : 用於取出ECG影像波型，適用於影像中背景雜訊與曲線顏色差異大，較不易失真。
> folder_path 為ECG原始影像路徑，save_path為儲存之目標路徑，color為目標波型顏色，共可取出藍色、黑色、灰色、白色、紅色、橘色、黃色、綠色、紫色九種色彩。
>
```python
def process_image_inrange(folder_path, save_path, color)
```
> 用途 : 用於取出ECG影像波型，適用於影像中背景雜訊與曲線顏色差異大，較不易失真。
> folder_path 為ECG原始影像路徑，save_path為儲存之目標路徑，color為目標波型顏色。
>


