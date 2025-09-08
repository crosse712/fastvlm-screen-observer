try:
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Selenium not installed - demo automation disabled")

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    print("PyAutoGUI not installed - automation features limited")

import time
import asyncio

class BrowserAutomation:
    def __init__(self):
        self.driver = None
        if PYAUTOGUI_AVAILABLE:
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.5
    
    def initialize_driver(self):
        if not SELENIUM_AVAILABLE:
            print("Selenium not available - cannot initialize driver")
            return
            
        try:
            chrome_options = Options()
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            self.driver.set_window_size(1280, 720)
            self.driver.set_window_position(100, 100)
            
        except Exception as e:
            print(f"Driver initialization error: {e}")
            self.driver = None
    
    async def run_demo(self, url: str, text_to_type: str):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._run_demo_sync, url, text_to_type)
    
    def _run_demo_sync(self, url: str, text_to_type: str):
        if not SELENIUM_AVAILABLE:
            print(f"Demo mode: Would open {url} and type '{text_to_type}'")
            time.sleep(2)
            return
            
        try:
            if self.driver is None:
                self.initialize_driver()
            
            if self.driver:
                self.driver.get(url)
                
                time.sleep(2)
                
                try:
                    search_box = self.driver.find_element(By.TAG_NAME, "input")
                    search_box.click()
                    search_box.send_keys(text_to_type)
                except:
                    body = self.driver.find_element(By.TAG_NAME, "body")
                    body.click()
                    body.send_keys(text_to_type)
                
                time.sleep(1)
                
                if PYAUTOGUI_AVAILABLE:
                    original_window = pyautogui.getActiveWindow()
                    if original_window:
                        original_window.activate()
                
                time.sleep(5)
                
                self.driver.quit()
                self.driver = None
                
        except Exception as e:
            print(f"Demo execution error: {e}")
            if self.driver:
                self.driver.quit()
                self.driver = None
    
    def cleanup(self):
        if self.driver:
            self.driver.quit()
            self.driver = None