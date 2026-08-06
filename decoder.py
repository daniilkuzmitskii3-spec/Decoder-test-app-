import traceback
import sys

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext
    import pyautogui
    import keyboard
    import threading
    import time
    from PIL import ImageGrab, Image
    import cv2
    import numpy as np
    import json
    import os
    import warnings
    warnings.filterwarnings('ignore')
    
    import easyocr
    print("Импорт успешен")
except Exception as e:
    print(f"Ошибка импорта: {e}")
    traceback.print_exc()
    input("Нажмите Enter...")
    sys.exit(1)

pyautogui.PAUSE = 0.05
pyautogui.FAILSAFE = True

MAX_RETRIES = 5
REQUIRED_DIGITS = 10
VERIFICATION_PASSES = 2
SETTINGS_FILE = 'decoder_settings.json'

DECODE_TABLE = {
    'A': '1', 'G': '2', 'S': '3', 'B': '4', 'V': '5',
    'H': '6', 'J': '7', 'Y': '8', 'T': '9', 'K': '0',
    '%': '5', '$': '2', '@': '7', '?': '9', '>': '1', '<': '0',
}

ALLOWED = set(DECODE_TABLE.keys())

def filter_chars(text):
    return ''.join(c for c in text if c in ALLOWED or c.isspace())

def decode(text):
    text = text.upper().strip()
    result = []
    
    for i, ch in enumerate(text):
        if ch in ('*', '#'):
            continue
        
        if ch == '$':
            result.append('2')
        elif ch == 'S':
            result.append('3')
        elif ch in DECODE_TABLE:
            result.append(DECODE_TABLE[ch])
        else:
            result.append(ch)
    
    return ''.join(result)

def extract_digits(text):
    return ''.join(c for c in text if c.isdigit())

def compare_digits(d1, d2):
    if len(d1) != len(d2):
        return False, f"Разная длина: {len(d1)} vs {len(d2)}"
    
    differences = []
    for i, (c1, c2) in enumerate(zip(d1, d2)):
        if c1 != c2:
            differences.append(f"Поз.{i+1}: {c1} vs {c2}")
    
    if differences:
        return False, ", ".join(differences)
    return True, "Идентичны"

def preprocess(img_array, strength='normal'):
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    if strength == 'light':
        scale = 2
        denoise_h = 10
        clip_limit = 2.0
        block_size = 15
        c_value = 5
    elif strength == 'strong':
        scale = 5
        denoise_h = 30
        clip_limit = 5.0
        block_size = 7
        c_value = 1
    else:
        scale = 4
        denoise_h = 20
        clip_limit = 3.0
        block_size = 11
        c_value = 2
    
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bitwise_not(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=denoise_h)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block_size, c_value)
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    kernel_dilate = np.ones((2,2), np.uint8)
    binary = cv2.dilate(binary, kernel_dilate, iterations=1)
    
    return binary

def enhance_for_problem_chars(img_array):
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    versions = []
    
    v1 = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    v1 = cv2.equalizeHist(v1)
    versions.append(v1)
    
    v2 = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    v2 = cv2.adaptiveThreshold(v2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY, 11, 2)
    versions.append(v2)
    
    v3 = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    v3 = cv2.Canny(v3, 50, 150)
    versions.append(v3)
    
    combined = versions[0]
    for v in versions[1:]:
        combined = cv2.addWeighted(combined, 0.7, v, 0.3, 0)
    
    return combined

class DecoderApp:
    def __init__(self, root):
        self.root = root
        root.title("Декодер + Автоклик (Автосохранение)")
        root.geometry("950x850")
        
        # Инициализация всех переменных ДО загрузки настроек
        self.selecting = False
        self.zones = {}
        self.confirm_zone = None
        self.fixed_zone = None
        self.last_zone = None
        self.auto_click_enabled = tk.BooleanVar(value=False)
        self.auto_click_timer = None
        self.click_cancelled = False
        self.retry_count = 0
        self.verification_results = []
        self.enhance_mode = tk.StringVar(value="normal")
        
        # Загружаем настройки (только данные, без UI)
        self.load_settings_data()
        
        print("Загрузка EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=False, 
                                     model_storage_directory='./models',
                                     download_enabled=True)
        print("Готов!")
        
        # Создаем интерфейс
        self.setup_ui()
        
        # Привязываем сохранение при закрытии окна
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_settings_data(self):
        """Загружает только данные настроек без UI"""
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                self.fixed_zone = settings.get('fixed_zone')
                self.zones = settings.get('zones', {})
                self.confirm_zone = settings.get('confirm_zone')
                self.last_zone = settings.get('last_zone')
                
                # Загружаем режим улучшения
                enhance = settings.get('enhance_mode', 'normal')
                if hasattr(self, 'enhance_mode'):
                    self.enhance_mode.set(enhance)
                
                # Загружаем настройку автоклика
                auto_click = settings.get('auto_click_enabled', False)
                if hasattr(self, 'auto_click_enabled'):
                    self.auto_click_enabled.set(auto_click)
                
                print("Настройки загружены успешно")
                return True
            else:
                print("Файл настроек не найден")
                return False
                
        except Exception as e:
            print(f"Ошибка загрузки настроек: {e}")
            return False
    
    def setup_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок
        header_frame = ttk.Frame(main)
        header_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(header_frame, text="ДЕКОДЕР + АВТОКЛИК (АВТОСОХРАНЕНИЕ)", 
                 font=('Arial', 14, 'bold')).pack(side=tk.LEFT)
        
        self.save_status_label = ttk.Label(header_frame, text="", foreground='green')
        self.save_status_label.pack(side=tk.RIGHT)
        self.update_save_status()
        
        # Настройки различения
        settings_frame = ttk.LabelFrame(main, text="Настройки различения символов", padding=5)
        settings_frame.pack(fill=tk.X, pady=5)
        
        settings_row = ttk.Frame(settings_frame)
        settings_row.pack()
        
        ttk.Label(settings_row, text="Режим:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(settings_row, text="Нормальный", variable=self.enhance_mode, 
                       value="normal").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(settings_row, text="Агрессивный", variable=self.enhance_mode, 
                       value="aggressive").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(settings_row, text="Максимальный", variable=self.enhance_mode, 
                       value="maximum").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(settings_row, text="📊 Тест различения", 
                  command=self.test_discrimination).pack(side=tk.LEFT, padx=20)
        
        # Кнопки управления сохранением
        save_frame = ttk.Frame(settings_frame)
        save_frame.pack(pady=5)
        ttk.Button(save_frame, text="💾 Сохранить сейчас", 
                  command=self.save_all_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(save_frame, text="📂 Загрузить настройки", 
                  command=self.load_all_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(save_frame, text="🗑️ Сбросить настройки", 
                  command=self.reset_settings).pack(side=tk.LEFT, padx=5)
        
        # Кнопки OCR
        ocr_frame = ttk.LabelFrame(main, text="Распознавание текста", padding=5)
        ocr_frame.pack(fill=tk.X, pady=5)
        
        ocr_btn = ttk.Frame(ocr_frame)
        ocr_btn.pack()
        ttk.Button(ocr_btn, text="F4 - ВЫДЕЛИТЬ", command=self.select_area).pack(side=tk.LEFT, padx=2)
        ttk.Button(ocr_btn, text="f5 - ФИКС. ЗОНА", command=self.set_fixed_zone).pack(side=tk.LEFT, padx=2)
        ttk.Button(ocr_btn, text="f2 - ДЕКОД ФИКС.", command=self.decode_fixed).pack(side=tk.LEFT, padx=2)
        ttk.Button(ocr_btn, text="ОЧИСТИТЬ", command=lambda: self.text_widget.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=2)
        ttk.Button(ocr_btn, text="🔄 ПЕРЕПРОВЕРИТЬ", command=self.recheck_last).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(ocr_btn, text="Автоклик после OCR", variable=self.auto_click_enabled).pack(side=tk.LEFT, padx=10)
        
        # Информация о фикс. зоне
        zone_info_frame = ttk.Frame(ocr_frame)
        zone_info_frame.pack(pady=5, fill=tk.X)
        
        fix_text = "Фикс. зона: не установлена"
        fix_color = 'red'
        if self.fixed_zone:
            x1, y1, x2, y2 = self.fixed_zone
            fix_text = f"Фикс. зона: {x2-x1}x{y2-y1} пикс. [СОХРАНЕНА]"
            fix_color = 'green'
        
        self.fixed_label = ttk.Label(zone_info_frame, text=fix_text, 
                                     foreground=fix_color, font=('Arial', 10, 'bold'))
        self.fixed_label.pack(side=tk.LEFT, padx=5)
        
        # Информация о проверке
        self.check_frame = ttk.Frame(ocr_frame)
        self.check_frame.pack(pady=2)
        self.digits_count_label = ttk.Label(self.check_frame, text="Цифр: 0/10", font=('Arial', 10, 'bold'))
        self.digits_count_label.pack(side=tk.LEFT, padx=5)
        self.verification_label = ttk.Label(self.check_frame, text="Проверок: 0/2", font=('Arial', 10))
        self.verification_label.pack(side=tk.LEFT, padx=5)
        self.discrimination_label = ttk.Label(self.check_frame, text="", foreground='purple')
        self.discrimination_label.pack(side=tk.LEFT, padx=5)
        
        # Поле текста
        text_frame = ttk.LabelFrame(main, text="Распознанный текст", padding=5)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.text_widget = scrolledtext.ScrolledText(text_frame, font=('Consolas', 14), height=8)
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Поле сравнения
        compare_frame = ttk.LabelFrame(main, text="Анализ символов", padding=5)
        compare_frame.pack(fill=tk.X, pady=5)
        
        self.compare_widget = scrolledtext.ScrolledText(compare_frame, font=('Consolas', 10), height=5)
        self.compare_widget.pack(fill=tk.BOTH)
        
        self.timer_label = ttk.Label(text_frame, text="", foreground='blue', font=('Arial', 12, 'bold'))
        self.timer_label.pack()
        
        # Зоны для кликов
        zone_frame = ttk.LabelFrame(main, text="Зоны для кликов", padding=5)
        zone_frame.pack(fill=tk.X, pady=5)
        
        btn_row = ttk.Frame(zone_frame)
        btn_row.pack()
        for i in range(10):
            d = str(i)
            ttk.Button(btn_row, text=d, width=3, command=lambda d=d: self.create_zone(d)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="ПОДТВ", width=7, command=lambda: self.create_zone('confirm')).pack(side=tk.LEFT, padx=10)
        
        self.zone_listbox = tk.Listbox(zone_frame, height=3, font=('Consolas', 10))
        self.zone_listbox.pack(fill=tk.X, pady=5)
        self.update_zone_list()
        
        ttk.Button(main, text="▶ АВТОКЛИК (МЕДЛЕННО)", command=self.auto_click).pack(pady=5, fill=tk.X)
        ttk.Button(main, text="■ СТОП", command=self.stop_auto_click).pack(pady=5, fill=tk.X)
        
        # Лог
        self.log_widget = scrolledtext.ScrolledText(main, height=6, font=('Consolas', 9))
        self.log_widget.pack(fill=tk.BOTH)
        
        # Горячие клавиши
        try:
            keyboard.add_hotkey('f4', self.select_area)
            keyboard.add_hotkey('f5', self.set_fixed_zone)
            keyboard.add_hotkey('f2', self.decode_fixed)
            self.log("F4/f5/f2 зарегистрированы")
        except:
            self.log("Запустите от админа для горячих клавиш")
        
        self.log("Готов! Автосохранение активировано")
        if self.fixed_zone:
            self.log("✓ Загружена сохраненная фикс. зона")
        if self.zones:
            self.log(f"✓ Загружено {len(self.zones)} зон для кликов")
    
    def update_save_status(self):
        """Обновляет статус сохранения"""
        if os.path.exists(SETTINGS_FILE):
            mod_time = os.path.getmtime(SETTINGS_FILE)
            time_str = time.strftime("%H:%M:%S", time.localtime(mod_time))
            self.save_status_label.config(
                text=f"🟢 Настройки сохранены ({time_str})", 
                foreground='green'
            )
        else:
            self.save_status_label.config(
                text="🔴 Настройки не сохранены", 
                foreground='red'
            )
    
    def save_all_settings(self):
        """Сохраняет все настройки включая фикс зону"""
        try:
            settings = {
                'fixed_zone': self.fixed_zone,
                'zones': self.zones,
                'confirm_zone': self.confirm_zone,
                'enhance_mode': self.enhance_mode.get(),
                'auto_click_enabled': self.auto_click_enabled.get(),
                'window_geometry': self.root.geometry(),
                'last_zone': getattr(self, 'last_zone', None)
            }
            
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            
            self.update_save_status()
            self.log("💾 Все настройки сохранены (включая фикс. зону)")
            
        except Exception as e:
            self.log(f"❌ Ошибка сохранения: {e}")
    
    def load_all_settings(self):
        """Загружает все настройки и обновляет UI"""
        if self.load_settings_data():
            self.update_zone_list()
            if self.fixed_zone:
                x1, y1, x2, y2 = self.fixed_zone
                self.fixed_label.config(
                    text=f"Фикс. зона: {x2-x1}x{y2-y1} пикс. [СОХРАНЕНА]", 
                    foreground='green'
                )
            self.log("📂 Настройки загружены")
        else:
            self.log("📝 Файл настроек не найден")
    
    def reset_settings(self):
        """Сбрасывает все настройки"""
        if os.path.exists(SETTINGS_FILE):
            os.remove(SETTINGS_FILE)
        
        self.fixed_zone = None
        self.zones = {}
        self.confirm_zone = None
        self.enhance_mode.set('normal')
        self.auto_click_enabled.set(False)
        
        self.fixed_label.config(text="Фикс. зона: не установлена", foreground='red')
        self.update_zone_list()
        self.update_save_status()
        self.log("🗑️ Все настройки сброшены")
    
    def on_closing(self):
        """Вызывается при закрытии окна"""
        self.log("🔄 Закрытие программы...")
        self.save_all_settings()
        self.log("✅ Настройки сохранены перед выходом")
        self.root.destroy()
    
    def log(self, msg):
        t = time.strftime("%H:%M:%S")
        self.log_widget.insert('end', f"[{t}] {msg}\n")
        self.log_widget.see('end')
        self.root.update()
    
    def analyze_symbols(self, img_array):
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        gray = cv2.resize(gray, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        analysis = {
            'total_symbols': len(contours),
            's_like': 0,
            'dollar_like': 0,
            'details': []
        }
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            if 0.3 < aspect_ratio < 1.5 and w > 10 and h > 15:
                roi = gray[y:y+h, x:x+w]
                edges = cv2.Canny(roi, 50, 150)
                vertical_lines = cv2.HoughLinesP(edges, 1, np.pi/2, threshold=20, 
                                                minLineLength=h//2, maxLineGap=5)
                
                if vertical_lines is not None and len(vertical_lines) >= 2:
                    analysis['dollar_like'] += 1
                    analysis['details'].append(f"Символ {len(analysis['details'])+1}: похож на $")
                else:
                    analysis['s_like'] += 1
                    analysis['details'].append(f"Символ {len(analysis['details'])+1}: похож на S")
        
        return analysis
    
    def test_discrimination(self):
        if not hasattr(self, 'last_zone') or not self.last_zone:
            self.log("Сначала выделите область!")
            return
        
        x1, y1, x2, y2 = self.last_zone
        self.log("🔬 Анализ символов...")
        
        img = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        arr = np.array(img)
        
        analysis = self.analyze_symbols(arr)
        
        self.compare_widget.delete(1.0, tk.END)
        self.compare_widget.insert(1.0, 
            f"АНАЛИЗ СИМВОЛОВ:\n"
            f"Всего символов: {analysis['total_symbols']}\n"
            f"Похожих на $: {analysis['dollar_like']}\n"
            f"Похожих на S: {analysis['s_like']}\n"
            f"Детали:\n" + "\n".join(analysis['details'])
        )
        
        self.discrimination_label.config(
            text=f"$: {analysis['dollar_like']} | S: {analysis['s_like']}",
            foreground='blue'
        )
    
    def select_area(self):
        if self.selecting:
            return
        self.selecting = True
        self.root.iconify()
        time.sleep(0.3)
        
        win = tk.Toplevel(self.root)
        win.attributes('-fullscreen', True, '-alpha', 0.3, '-topmost', True)
        win.config(cursor='cross', bg='gray')
        canvas = tk.Canvas(win, bg='gray')
        canvas.pack(fill=tk.BOTH, expand=True)
        tk.Label(win, text="ВЫДЕЛИТЕ ОБЛАСТЬ", font=('Arial', 18, 'bold'), fg='white', bg='gray').place(relx=0.5, rely=0.05, anchor=tk.CENTER)
        
        start = [0, 0]
        rect = [None]
        
        def down(e):
            start[0], start[1] = e.x_root, e.y_root
            rect[0] = canvas.create_rectangle(e.x_root, e.y_root, e.x_root, e.y_root, outline='red', width=3)
        
        def move(e):
            if rect[0]:
                canvas.coords(rect[0], start[0], start[1], e.x_root, e.y_root)
        
        def up(e):
            if rect[0]:
                x1, y1 = min(start[0], e.x_root), min(start[1], e.y_root)
                x2, y2 = max(start[0], e.x_root), max(start[1], e.y_root)
                win.destroy()
                self.selecting = False
                self.root.deiconify()
                self.root.lift()
                if x2-x1 > 20 and y2-y1 > 20:
                    self.last_zone = (x1, y1, x2, y2)
                    self.retry_count = 0
                    self.verification_results = []
                    threading.Thread(target=self.ocr_with_double_verification, 
                                   args=(x1, y1, x2, y2, True), daemon=True).start()
        
        canvas.bind('<ButtonPress-1>', down)
        canvas.bind('<B1-Motion>', move)
        canvas.bind('<ButtonRelease-1>', up)
        win.bind('<Escape>', lambda e: [win.destroy(), setattr(self, 'selecting', False), 
                                        self.root.deiconify(), self.root.lift()])
    
    def set_fixed_zone(self):
        if self.selecting:
            return
        self.selecting = True
        self.root.iconify()
        time.sleep(0.3)
        
        win = tk.Toplevel(self.root)
        win.attributes('-fullscreen', True, '-alpha', 0.3, '-topmost', True)
        win.config(cursor='cross', bg='gray')
        canvas = tk.Canvas(win, bg='gray')
        canvas.pack(fill=tk.BOTH, expand=True)
        tk.Label(win, text="ВЫДЕЛИТЕ ФИКСИРОВАННУЮ ЗОНУ", font=('Arial', 18, 'bold'), 
                fg='white', bg='gray').place(relx=0.5, rely=0.05, anchor=tk.CENTER)
        
        start = [0, 0]
        rect = [None]
        
        def down(e):
            start[0], start[1] = e.x_root, e.y_root
            rect[0] = canvas.create_rectangle(e.x_root, e.y_root, e.x_root, e.y_root, 
                                             outline='blue', width=3)
        
        def move(e):
            if rect[0]:
                canvas.coords(rect[0], start[0], start[1], e.x_root, e.y_root)
        
        def up(e):
            if rect[0]:
                x1, y1 = min(start[0], e.x_root), min(start[1], e.y_root)
                x2, y2 = max(start[0], e.x_root), max(start[1], e.y_root)
                win.destroy()
                self.selecting = False
                self.root.deiconify()
                self.root.lift()
                if x2-x1 > 20 and y2-y1 > 20:
                    self.fixed_zone = (x1, y1, x2, y2)
                    self.fixed_label.config(
                        text=f"Фикс. зона: {x2-x1}x{y2-y1} пикс. [АВТОСОХРАНЕНИЕ]", 
                        foreground='green'
                    )
                    self.log(f"✅ Фиксированная зона установлена и сохранена")
                    self.save_all_settings()
        
        canvas.bind('<ButtonPress-1>', down)
        canvas.bind('<B1-Motion>', move)
        canvas.bind('<ButtonRelease-1>', up)
        win.bind('<Escape>', lambda e: [win.destroy(), setattr(self, 'selecting', False), 
                                        self.root.deiconify(), self.root.lift()])
    
    def decode_fixed(self):
        if not self.fixed_zone:
            self.log("Фиксированная зона не установлена!")
            return
        
        x1, y1, x2, y2 = self.fixed_zone
        self.last_zone = self.fixed_zone
        self.retry_count = 0
        self.verification_results = []
        self.log("Декодирование фикс. зоны...")
        threading.Thread(target=self.ocr_with_double_verification, 
                        args=(x1, y1, x2, y2, True), daemon=True).start()
    
    def recheck_last(self):
        if hasattr(self, 'last_zone') and self.last_zone:
            x1, y1, x2, y2 = self.last_zone
            self.retry_count = 0
            self.verification_results = []
            self.log("🔄 Принудительная перепроверка...")
            threading.Thread(target=self.ocr_with_double_verification, 
                           args=(x1, y1, x2, y2, True), daemon=True).start()
        else:
            self.log("Нет данных для перепроверки")
    
    def single_ocr_pass(self, x1, y1, x2, y2, method='normal'):
        img = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        arr = np.array(img)
        
        analysis = self.analyze_symbols(arr)
        mode = self.enhance_mode.get()
        
        if mode == "aggressive":
            processed = enhance_for_problem_chars(arr)
        elif mode == "maximum":
            processed_normal = preprocess(arr, 'normal')
            processed_special = enhance_for_problem_chars(arr)
            processed = cv2.addWeighted(processed_normal, 0.5, processed_special, 0.5, 0)
        else:
            processed = preprocess(arr, method)
        
        timestamp = time.strftime("%H%M%S")
        cv2.imwrite(f'debug_{method}_{timestamp}.png', processed)
        
        results = []
        
        res1 = self.reader.readtext(processed, detail=0)
        results.append(' '.join(res1).strip())
        
        if mode in ["aggressive", "maximum"]:
            inverted = cv2.bitwise_not(processed)
            res2 = self.reader.readtext(inverted, detail=0)
            results.append(' '.join(res2).strip())
        
        best_result = None
        best_score = 0
        
        for result in results:
            if result:
                filtered = filter_chars(result)
                decoded = decode(filtered)
                digits = extract_digits(decoded)
                
                score = 0
                if len(digits) == REQUIRED_DIGITS:
                    score += 100
                
                if '$' in result and '2' in digits:
                    score += 50
                if 'S' in result and '3' in digits:
                    score += 50
                
                if score > best_score:
                    best_score = score
                    best_result = {
                        'raw': result,
                        'filtered': filtered,
                        'decoded': decoded,
                        'digits': digits,
                        'method': method,
                        'success': len(digits) == REQUIRED_DIGITS,
                        'analysis': analysis
                    }
        
        return best_result
    
    def ocr_with_double_verification(self, x1, y1, x2, y2, auto_click=False):
        self.log("=" * 50)
        self.log("🔍 НАЧАЛО ДВОЙНОЙ ПРОВЕРКИ")
        
        self.compare_widget.delete(1.0, tk.END)
        
        verification_passes = []
        final_result = None
        
        self.log("📝 ПЕРВАЯ ПРОВЕРКА...")
        self.verification_label.config(text="Проверок: 1/2")
        
        for attempt in range(MAX_RETRIES):
            if attempt > 0:
                self.log(f"Первая проверка, попытка {attempt + 1}...")
                time.sleep(0.3)
            
            methods = ['normal', 'strong', 'light', 'normal', 'strong']
            result1 = self.single_ocr_pass(x1, y1, x2, y2, methods[attempt])
            
            if result1 and result1['success']:
                self.log(f"✅ Первая проверка: {result1['digits']}")
                if result1.get('analysis'):
                    self.log(f"   Анализ: $={result1['analysis']['dollar_like']}, S={result1['analysis']['s_like']}")
                verification_passes.append(result1)
                break
            elif result1:
                self.log(f"⚠️ Первая проверка: {result1['digits']} ({len(result1['digits'])}/10 цифр)")
        
        if not verification_passes:
            self.log("❌ Первая проверка не удалась")
            self.verification_label.config(text="Проверок: 0/2", foreground='red')
            return None
        
        self.log("📝 ВТОРАЯ ПРОВЕРКА (альтернативный метод)...")
        self.verification_label.config(text="Проверок: 2/2")
        time.sleep(0.3)
        
        for attempt in range(MAX_RETRIES):
            if attempt > 0:
                self.log(f"Вторая проверка, попытка {attempt + 1}...")
                time.sleep(0.3)
            
            methods = ['strong', 'light', 'normal', 'strong', 'light']
            result2 = self.single_ocr_pass(x1, y1, x2, y2, methods[attempt])
            
            if result2 and result2['success']:
                self.log(f"✅ Вторая проверка: {result2['digits']}")
                if result2.get('analysis'):
                    self.log(f"   Анализ: $={result2['analysis']['dollar_like']}, S={result2['analysis']['s_like']}")
                verification_passes.append(result2)
                break
            elif result2:
                self.log(f"⚠️ Вторая проверка: {result2['digits']} ({len(result2['digits'])}/10 цифр)")
        
        if len(verification_passes) >= 2:
            digits1 = verification_passes[0]['digits']
            digits2 = verification_passes[1]['digits']
            
            is_same, diff_info = compare_digits(digits1, digits2)
            
            analysis_text = []
            analysis_text.append(f"ПРОВЕРКА 1 [{verification_passes[0]['method']}]: {digits1}")
            if verification_passes[0].get('analysis'):
                a = verification_passes[0]['analysis']
                analysis_text.append(f"  Символов: {a['total_symbols']}, $: {a['dollar_like']}, S: {a['s_like']}")
            
            analysis_text.append(f"ПРОВЕРКА 2 [{verification_passes[1]['method']}]: {digits2}")
            if verification_passes[1].get('analysis'):
                a = verification_passes[1]['analysis']
                analysis_text.append(f"  Символов: {a['total_symbols']}, $: {a['dollar_like']}, S: {a['s_like']}")
            
            analysis_text.append(f"РЕЗУЛЬТАТ: {'✅ СОВПАДАЮТ' if is_same else '❌ РАЗЛИЧИЯ: ' + diff_info}")
            
            self.compare_widget.insert(1.0, "\n".join(analysis_text))
            
            if is_same:
                self.log(f"✅✅ ДВОЙНАЯ ПРОВЕРКА ПРОЙДЕНА: {digits1}")
                self.verification_label.config(text="Проверок: 2/2 ✅", foreground='green')
                final_result = digits1
            else:
                self.log(f"⚠️ Результаты различаются: {diff_info}")
                self.log("Проводим третью проверку...")
                self.verification_label.config(text="Проверок: 3/2 ⚠️", foreground='orange')
                
                time.sleep(0.3)
                result3 = self.single_ocr_pass(x1, y1, x2, y2, 'normal')
                
                if result3 and result3['success']:
                    digits3 = result3['digits']
                    self.compare_widget.insert('end', f"\nПРОВЕРКА 3 [normal]: {digits3}\n")
                    
                    if digits3 == digits1:
                        self.log(f"✅ Третья проверка подтвердила первый результат: {digits1}")
                        final_result = digits1
                    elif digits3 == digits2:
                        self.log(f"✅ Третья проверка подтвердила второй результат: {digits2}")
                        final_result = digits2
                    else:
                        self.log(f"⚠️ Все три результата разные! Используем первый: {digits1}")
                        final_result = digits1
        
        elif len(verification_passes) == 1:
            self.log("⚠️ Только одна проверка успешна")
            final_result = verification_passes[0]['digits']
        
        if final_result:
            self.text_widget.delete(1.0, tk.END)
            self.text_widget.insert(1.0, 
                f"=== ИТОГОВЫЙ РЕЗУЛЬТАТ ===\n"
                f"ЦИФРЫ: {final_result}\n"
                f"КОЛИЧЕСТВО: {len(final_result)}/{REQUIRED_DIGITS}\n"
                f"ПРОВЕРЕНО: {len(verification_passes)} раз(а)\n"
                f"СТАТУС: {'✅ ПОДТВЕРЖДЕНО' if len(verification_passes) >= 2 else '⚠️ ОДНА ПРОВЕРКА'}"
            )
            
            self.digits_count_label.config(
                text=f"Цифр: {len(final_result)}/{REQUIRED_DIGITS}",
                foreground='green' if len(final_result) == REQUIRED_DIGITS else 'orange'
            )
            
            if auto_click and self.auto_click_enabled.get():
                self.start_auto_click_timer(final_result)
            
            return final_result
        
        self.log("❌ Двойная проверка не дала результатов")
        self.verification_label.config(text="Проверок: 0/2 ❌", foreground='red')
        return None
    
    def start_auto_click_timer(self, digits):
        self.stop_auto_click()
        self.click_digits = digits
        self.click_cancelled = False
        
        if len(digits) != REQUIRED_DIGITS:
            self.log(f"⚠️ ВНИМАНИЕ: автоклик для {len(digits)} цифр вместо {REQUIRED_DIGITS}!")
        
        def countdown():
            for i in range(1, 0, -1):
                if self.click_cancelled:
                    self.timer_label.config(text="Отменено", foreground='red')
                    return
                self.timer_label.config(text=f"Автоклик через {i}...", foreground='blue')
                time.sleep(1)
            
            if not self.click_cancelled:
                self.timer_label.config(text="Выполняю клики...", foreground='green')
                self.do_clicks()
        
        self.auto_click_timer = threading.Thread(target=countdown, daemon=True)
        self.auto_click_timer.start()
    
    def stop_auto_click(self):
        self.click_cancelled = True
        self.timer_label.config(text="Остановлено")
        self.log("Автоклик остановлен")
    
    def do_clicks(self):
        if not hasattr(self, 'click_digits'):
            return
        
        digits = self.click_digits
        
        if len(digits) != REQUIRED_DIGITS:
            self.log(f"⚠️ Выполняется {len(digits)} кликов вместо {REQUIRED_DIGITS}")
        
        n = 0
        for ch in digits:
            if self.click_cancelled:
                self.log("Клики прерваны")
                return
                
            if ch.isdigit() and ch in self.zones:
                z = self.zones[ch]
                cx, cy = z['x']+z['width']//2, z['y']+z['height']//2
                
                pyautogui.moveTo(cx, cy, duration=0.1)
                time.sleep(0.05)
                
                pyautogui.mouseDown()
                time.sleep(0.01)
                pyautogui.mouseUp()
                
                n += 1
                self.log(f"Клик {n}/{len(digits)}: цифра {ch} в ({cx}, {cy})")
                
                time.sleep(0.1)
        
        if self.confirm_zone and not self.click_cancelled:
            time.sleep(0.15)
            z = self.confirm_zone
            cx, cy = z['x']+z['width']//2, z['y']+z['height']//2
            pyautogui.moveTo(cx, cy, duration=0.1)
            time.sleep(0.05)
            pyautogui.mouseDown()
            time.sleep(0.01)
            pyautogui.mouseUp()
            n += 1
            self.log(f"Клик подтверждения в ({cx}, {cy})")
        
        if not self.click_cancelled:
            self.timer_label.config(text=f"Готово! Кликов: {n}", foreground='green')
            self.log(f"Завершено! Всего кликов: {n}")
    
    def create_zone(self, digit):
        self.root.iconify()
        time.sleep(0.3)
        win = tk.Toplevel(self.root)
        win.attributes('-fullscreen', True, '-alpha', 0.3, '-topmost', True)
        win.config(cursor='cross', bg='gray')
        canvas = tk.Canvas(win, bg='gray')
        canvas.pack(fill=tk.BOTH, expand=True)
        label = "ПОДТВЕРЖДЕНИЕ" if digit == 'confirm' else f"Цифра {digit}"
        tk.Label(win, text=label, font=('Arial', 18, 'bold'), fg='white', bg='gray').place(
            relx=0.5, rely=0.05, anchor=tk.CENTER)
        
        start = [0, 0]
        rect = [None]
        
        def down(e):
            start[0], start[1] = e.x_root, e.y_root
            rect[0] = canvas.create_rectangle(e.x_root, e.y_root, e.x_root, e.y_root, 
                                             outline='green', width=3)
        
        def move(e):
            if rect[0]:
                canvas.coords(rect[0], start[0], start[1], e.x_root, e.y_root)
        
        def up(e):
            if rect[0]:
                x1, y1 = min(start[0], e.x_root), min(start[1], e.y_root)
                x2, y2 = max(start[0], e.x_root), max(start[1], e.y_root)
                win.destroy()
                self.root.deiconify()
                self.root.lift()
                if x2-x1 > 10 and y2-y1 > 10:
                    z = {'x': x1, 'y': y1, 'width': x2-x1, 'height': y2-y1}
                    if digit == 'confirm':
                        self.confirm_zone = z
                    else:
                        self.zones[digit] = z
                    self.log(f"✅ Зона {digit} создана и сохранена")
                    self.update_zone_list()
                    self.save_all_settings()
        
        canvas.bind('<ButtonPress-1>', down)
        canvas.bind('<B1-Motion>', move)
        canvas.bind('<ButtonRelease-1>', up)
        win.bind('<Escape>', lambda e: [win.destroy(), self.root.deiconify(), self.root.lift()])
    
    def update_zone_list(self):
        self.zone_listbox.delete(0, tk.END)
        for d in sorted(self.zones.keys(), key=int):
            self.zone_listbox.insert(tk.END, f"Цифра {d}")
        if self.confirm_zone:
            self.zone_listbox.insert(tk.END, "Подтверждение")
    
    def auto_click(self):
        text = self.text_widget.get(1.0, tk.END).strip()
        if 'ЦИФРЫ:' in text:
            digits_text = text.split('ЦИФРЫ:')[-1].strip()
            digits = extract_digits(digits_text)
        else:
            digits = extract_digits(text)
        
        if not digits:
            self.log("Нет цифр для автоклика")
            return
        
        if len(digits) != REQUIRED_DIGITS:
            self.log(f"⚠️ Найдено {len(digits)} цифр вместо {REQUIRED_DIGITS}")
        
        self.start_auto_click_timer(digits)
    
    def save_zones(self):
        """Совместимость со старым методом"""
        self.save_all_settings()
    
    def load_zones(self):
        """Совместимость со старым методом"""
        self.load_all_settings()

if __name__ == "__main__":
    print("=" * 50)
    print("ДЕКОДЕР + АВТОКЛИК (АВТОСОХРАНЕНИЕ)")
    print(f"Требуется цифр: {REQUIRED_DIGITS}")
    print(f"Файл настроек: {SETTINGS_FILE}")
    print("=" * 50)
    
    try:
        root = tk.Tk()
        app = DecoderApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        traceback.print_exc()
        input("Нажмите Enter для выхода...")