import time
import threading
class Timer:
    def __init__(self):
        self.start_time = None
        self.running = False

    def start(self):
        """Bắt đầu hiển thị thời gian liên tục."""
        self.start_time = time.time()
        self.running = True
        threading.Thread(target=self._display_time, daemon=True).start()

    def _display_time(self):
        """Hiển thị thời gian đã chạy mỗi giây trên một dòng."""
        while self.running:
            elapsed_time = time.time() - self.start_time
            print(f"\rTime: {elapsed_time:.2f} giây", end="", flush=True)  # Hiển thị trên cùng một dòng
            time.sleep(1)

    def stop(self):
        """Dừng hiển thị thời gian và trả về tổng thời gian."""
        self.running = False
        elapsed_time = time.time() - self.start_time
        print(f"\rTổng Time: {elapsed_time:.2f} giây!")  # In tổng thời gian
        return elapsed_time