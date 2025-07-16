import requests
import json
import gdown

# Thông tin API
url = "https://java.genadata.com/share"
access_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJ1c2VyX2NyZWRlbnRpYWxzIiwiZW1haWwiOiJ2dWRhbmcxNDAyMDVAZ21haWwuY29tIiwiZXhwIjoxNzUxMjY2NTUyLCJpYXQiOjE3NDg2NzQ1NTIsImlzcyI6IkdlbmlGYXN0LVNlYXJjaF9HbyIsIm5iZiI6MTc0ODY3NDU1MiwidXNlcl9pZCI6IjI1NCJ9.P2IpI7EGguF3DOuFjWwU4-R4TW80ShxLB_PTAsXVKIoXtyQEOR1M4C13vl0ae1AzGqA4Z22bcjbuyNQMyGrzb8MfO1y8GIBzt8uuQzrOuxd-9fT3G4X6Fh3MGJFJ9HHAtaXot57OGeevWsQv9FGZ1c-ROlaQVqJBIil5czRdeHemnwMn1er-saswmDx3UqodnfDjq1A4PjT0JkaYj560mXrkec1WKt-u14uhpx81QFIaX4CBYwlWNE-XdZr454mnjC70-bcQ2rjzC3BAzxFDbfY2dmAliQehTEG_X7KXpYHN_Ax8f7I2TZgQV79FuhO9xFbWZvSi_MK4cglOVJ8C0K3T9PmWICYEpVgJWxk57VFXKcax5fAcLq_DKLfvneTjpko1gXzK_iocmYayHGhxUlaHtzuPdsukrqUOgnJxzjLXj8rjUpjdwSAAkfiKJKcgl1yj8VuZwuPulbkct2qK_hRLG0ExPNX3LC_3xq_uPwupVKIzae6MYLFVqug6kFgtZFY3BBZojrcyiueS1XmT1gF4mIMaS-1SLLEnKcJggj7c_h-TI0QdE1kBmEa3mW41uYSv_knkev5wN9-Em6GVI4RupVO9WCyt-a3F1iSQGslmhX_YVS3Hio6E7P28JAX3pvwGCNhpD4YCs-Jdm1Bo-sDULXcqFJDxdor3KucOg_U"

# Cấu hình headers
headers = {
    "Authorization": f"Bearer {access_token}",
    "Accept": "application/json, text/plain, */*",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# Gửi yêu cầu GET
response = requests.get(url, headers=headers)

# Kiểm tra trạng thái và xử lý phản hồi
print("Status:", response.status_code)

if response.status_code == 200:
    try:
        data = response.json()
        # print(data)
        # Xử lý dữ liệu trả về
        if isinstance(data, list) and len(data) > 0:
            for item in data:
                # Lấy liên kết Google Drive
                file_url = item.get('filePath', '')
                file_id = item.get('fileId', '')

                if file_url and file_id:
                    print("\nTải file gốc từ Google Drive để kiểm tra nội dung...")
                    output = "E:/document/lab/DATN_ChatbotHUST/RAG/data_hdu/db.txt"
                    try:
                        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

                        # Đọc lại để kiểm tra (tuỳ chọn, nếu cần)
                        with open(output, 'r', encoding='utf-8') as f:
                            content = f.read()


                        # print("\nNội dung file gốc từ Google Drive:")
                        # Chia thành các dòng và hiển thị
                        lines = content.split('\r') 
                        for line in lines:
                            print(line.strip())

                    except Exception as e:
                        print(f"Lỗi khi tải hoặc đọc file từ Google Drive: {e}")
                        print("Hiển thị nội dung từ API thay thế:")
                        content = item.get('content', '').strip()
                        if content:
                            if content.startswith('\ufeff'):
                                content = content[1:]
                            lines = content.split('\r')
                            for line in lines:
                                print(line.strip())

        elif isinstance(data, list) and len(data) == 0:
            print("Không có dữ liệu chia sẻ nào.")
        else:
            print("Phản hồi không phải là danh sách, có thể là JSON khác:", data)
    except json.JSONDecodeError:
        print("Lỗi: Phản hồi không phải là JSON hợp lệ.")
else:
    print(f"Lỗi: Trạng thái {response.status_code} - {response.text}")