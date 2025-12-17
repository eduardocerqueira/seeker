#date: 2025-12-17T17:01:55Z
#url: https://api.github.com/gists/a224872a98c348a77de92eddec61545d
#owner: https://api.github.com/users/bailipaobu-lgtm

"""
豆包图片分类管理系统 - 完整优化版 v3.4
新增: 手动API配置功能

作者: AI Assistant  
版本: 3.4 (手动API配置版)
"""

import customtkinter as ctk
from PIL import Image
import os
import base64
import requests
import re
import json
import threading
import shutil
import time
import hashlib
import pickle
import concurrent.futures
from pathlib import Path
from tkinter import messagebox, filedialog

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


def center_window(dialog, parent):
    """让对话框在父窗口中心显示 - 无闪烁版本"""
    dialog.withdraw()
    dialog.update_idletasks()
    
    parent_x = parent.winfo_x()
    parent_y = parent.winfo_y()
    parent_width = parent.winfo_width()
    parent_height = parent.winfo_height()
    
    dialog_width = dialog.winfo_reqwidth()
    dialog_height = dialog.winfo_reqheight()
    
    x = parent_x + (parent_width - dialog_width) // 2
    y = parent_y + (parent_height - dialog_height) // 2
    
    x = max(0, x)
    y = max(0, y)
    
    dialog.geometry(f"+{x}+{y}")
    dialog.deiconify()


class APIConfigDialog(ctk.CTkToplevel):
    """🔧 API配置对话框"""
    
    def __init__(self, parent, current_config):
        super().__init__(parent)
        
        self.title("🔧 API配置")
        self.geometry("600x500")
        self.transient(parent)
        self.grab_set()
        
        self.current_config = current_config
        self.result = None
        
        self.setup_ui()
        center_window(self, parent)
    
    def setup_ui(self):
        # 标题区域
        header = ctk.CTkFrame(self, height=80, fg_color=("#3b8ed0", "#1f6aa5"))
        header.pack(fill="x")
        header.pack_propagate(False)
        
        ctk.CTkLabel(
            header,
            text="🔧 豆包API配置",
            font=("Arial", 20, "bold"),
            text_color="white"
        ).pack(pady=(15, 5))
        
        ctk.CTkLabel(
            header,
            text="配置豆包大模型API访问凭证",
            font=("Arial", 11),
            text_color="white"
        ).pack()
        
        # 配置表单
        form_frame = ctk.CTkFrame(
            self,
            corner_radius=10,
            fg_color=("#f5f5f5", "#2b2b2b")
        )
        form_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # API Key
        ctk.CTkLabel(
            form_frame,
            text="🔑 API Key",
            font=("Arial", 14, "bold"),
            anchor="w"
        ).pack(pady=(20, 5), padx=20, anchor="w")
        
        ctk.CTkLabel(
            form_frame,
            text="在豆包开放平台获取你的API密钥",
            font=("Arial", 11),
            text_color="gray",
            anchor="w"
        ).pack(padx=20, anchor="w")
        
        self.api_key_entry = ctk.CTkEntry(
            form_frame,
            placeholder_text="请输入API Key",
            width=540,
            height=40,
            font=("Arial", 13)
        )
        self.api_key_entry.pack(pady=10, padx=20)
        
        # 预填充当前配置
        if self.current_config.get('api_key'):
            self.api_key_entry.insert(0, self.current_config['api_key'])
        
        # Model ID
        ctk.CTkLabel(
            form_frame,
            text="🤖 模型ID",
            font=("Arial", 14, "bold"),
            anchor="w"
        ).pack(pady=(20, 5), padx=20, anchor="w")
        
        ctk.CTkLabel(
            form_frame,
            text="选择或输入豆包视觉模型ID",
            font=("Arial", 11),
            text_color="gray",
            anchor="w"
        ).pack(padx=20, anchor="w")
        
        # 模型选择
        model_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        model_frame.pack(pady=10, padx=20, fill="x")
        
        self.model_var = ctk.StringVar(
            value=self.current_config.get('model_id', 'doubao-1-5-vision-pro-32k-250115')
        )
        
        models = [
            ("豆包视觉Pro (推荐)", "doubao-1-5-vision-pro-32k-250115"),
            ("豆包视觉标准", "doubao-vision-standard"),
            ("自定义模型ID", "custom")
        ]
        
        for idx, (label, value) in enumerate(models):
            ctk.CTkRadioButton(
                model_frame,
                text=label,
                variable=self.model_var,
                value=value,
                font=("Arial", 13),
                command=self.on_model_select
            ).pack(anchor="w", pady=5)
        
        # 自定义模型ID输入框
        self.custom_model_entry = ctk.CTkEntry(
            form_frame,
            placeholder_text="输入自定义模型ID",
            width=540,
            height=40,
            font=("Arial", 13),
            state="disabled"
        )
        self.custom_model_entry.pack(pady=5, padx=20)
        
        # 帮助信息
        help_frame = ctk.CTkFrame(
            form_frame,
            fg_color=("#e3f2fd", "#1e3a5f"),
            corner_radius=8
        )
        help_frame.pack(pady=20, padx=20, fill="x")
        
        ctk.CTkLabel(
            help_frame,
            text="💡 如何获取API Key?\n"
                 "1. 访问: https://console.volcengine.com/ark\n"
                 "2. 注册/登录火山引擎账号\n"
                 "3. 创建API Key并复制到上方输入框",
            font=("Arial", 11),
            justify="left",
            anchor="w"
        ).pack(pady=10, padx=15)
        
        # 按钮区域
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=20)
        
        ctk.CTkButton(
            btn_frame,
            text="✅ 保存配置",
            command=self.save_config,
            width=180,
            height=45,
            font=("Arial", 14, "bold"),
            fg_color="#4caf50",
            hover_color="#388e3c"
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            btn_frame,
            text="🧪 测试连接",
            command=self.test_connection,
            width=180,
            height=45,
            font=("Arial", 14, "bold"),
            fg_color="#2196f3",
            hover_color="#1976d2"
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            btn_frame,
            text="❌ 取消",
            command=self.destroy,
            width=180,
            height=45,
            font=("Arial", 14, "bold"),
            fg_color="gray",
            hover_color="#616161"
        ).pack(side="left", padx=10)
    
    def on_model_select(self):
        """模型选择回调"""
        if self.model_var.get() == "custom":
            self.custom_model_entry.configure(state="normal")
        else:
            self.custom_model_entry.configure(state="disabled")
    
    def save_config(self):
        """保存配置"""
        api_key = self.api_key_entry.get().strip()
        
        if not api_key:
            messagebox.showerror("错误", "❌ API Key不能为空!")
            return
        
        # 获取模型ID
        model_id = self.model_var.get()
        if model_id == "custom":
            model_id = self.custom_model_entry.get().strip()
            if not model_id:
                messagebox.showerror("错误", "❌ 请输入自定义模型ID!")
                return
        
        self.result = {
            'api_key': api_key,
            'model_id': model_id
        }
        
        messagebox.showinfo("成功", "✅ 配置已保存!")
        self.destroy()
    
    def test_connection(self):
        """测试API连接"""
        api_key = self.api_key_entry.get().strip()
        
        if not api_key:
            messagebox.showerror("错误", "❌ 请先输入API Key!")
            return
        
        model_id = self.model_var.get()
        if model_id == "custom":
            model_id = self.custom_model_entry.get().strip()
            if not model_id:
                messagebox.showerror("错误", "❌ 请输入自定义模型ID!")
                return
        
        # 显示测试提示
        test_dialog = ctk.CTkToplevel(self)
        test_dialog.title("测试中...")
        test_dialog.geometry("300x100")
        test_dialog.transient(self)
        
        ctk.CTkLabel(
            test_dialog,
            text="🔄 正在测试API连接...",
            font=("Arial", 14)
        ).pack(expand=True)
        
        center_window(test_dialog, self)
        
        # 后台测试
        def test_worker():
            try:
                url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                payload = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": "测试"}]
                }
                
                response = requests.post(url, headers=headers, json=payload, timeout=10)
                
                test_dialog.destroy()
                
                if response.status_code == 200:
                    messagebox.showinfo(
                        "测试成功",
                        "✅ API连接正常!\n\n"
                        f"模型: {model_id}\n"
                        "可以正常使用"
                    )
                elif response.status_code == 401:
                    messagebox.showerror("测试失败", "❌ API Key无效,请检查!")
                elif response.status_code == 404:
                    messagebox.showerror("测试失败", "❌ 模型ID不存在,请检查!")
                else:
                    messagebox.showerror(
                        "测试失败",
                        f"❌ 连接失败!\n\n"
                        f"状态码: {response.status_code}\n"
                        f"错误信息: {response.text[:100]}"
                    )
                    
            except requests.exceptions.Timeout:
                test_dialog.destroy()
                messagebox.showerror("测试失败", "❌ 连接超时,请检查网络!")
            except Exception as e:
                test_dialog.destroy()
                messagebox.showerror("测试失败", f"❌ 测试失败:\n{str(e)}")
        
        threading.Thread(target=test_worker, daemon=True).start()


class DoubaoDetector:
    """豆包AI检测器 - 支持动态配置"""
    
    def __init__(self, api_key=None, model_id=None):
        self.api_key = api_key
        self.model_id = model_id or "doubao-1-5-vision-pro-32k-250115"
        self.base_url = "https://ark.cn-beijing.volces.com/api/v3"
        
        self.cache_file = "detection_cache.pkl"
        self.cache = self.load_cache()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def update_config(self, api_key, model_id):
        """更新API配置"""
        self.api_key = api_key
        self.model_id = model_id
    
    def is_configured(self):
        """检查是否已配置"""
        return bool(self.api_key)
    
    def load_cache(self):
        """加载本地缓存"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    print(f"✅ 已加载缓存: {len(cache)} 条记录")
                    return cache
        except Exception as e:
            print(f"⚠️ 缓存加载失败: {e}")
        return {}
    
    def save_cache(self):
        """保存缓存到本地"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            print(f"💾 缓存已保存: {len(self.cache)} 条")
        except Exception as e:
            print(f"❌ 缓存保存失败: {e}")
    
    def get_image_hash(self, image_path):
        """计算图片MD5哈希值"""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None
    
    def detect_image(self, image_path):
        """检测单张图片(带缓存+重试)"""
        if not self.is_configured():
            return {
                'category': '检测失败',
                'confidence': 0,
                'success': False,
                'error': 'API未配置'
            }
        
        img_hash = self.get_image_hash(image_path)
        if img_hash and img_hash in self.cache:
            self.cache_hits += 1
            return self.cache[img_hash]
        
        self.cache_misses += 1
        result = self._detect_with_retry(image_path, max_retries=3)
        
        if result['success'] and img_hash:
            self.cache[img_hash] = result
        
        return result
    
    def _detect_with_retry(self, image_path, max_retries=3):
        """带重试的API调用"""
        for attempt in range(max_retries):
            try:
                with open(image_path, 'rb') as f:
                    img_base64 = base64.b64encode(f.read()).decode('utf-8')
                
                prompt = """请判断这张产品图片属于以下哪种类型:
【材质图】- 展示材料、质感、构造细节
【尺寸图】- 包含明确尺寸数字标注
【场景图】- 产品在真实生活环境中

返回格式:
分类: [类型]
置信度: [0-1数值]"""
                
                payload = {
                    "model": self.model_id,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }]
                }
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"⚠️ API限流,等待{wait_time}秒... (尝试 {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code in [400, 401, 403, 413]:
                    print(f"❌ 客户端错误 {response.status_code},跳过重试")
                    break
                
                if response.status_code == 200:
                    result_text = response.json()['choices'][0]['message']['content']
                    
                    match = re.search(r'分类[:：]\s*([^\n]+)', result_text)
                    category = '未分类'
                    if match:
                        for cat in ['材质图', '尺寸图', '场景图', '其他']:
                            if cat in match.group(1):
                                category = cat
                                break
                    
                    confidence = 0.5
                    match = re.search(r'置信度[:：]\s*(\d+\.?\d*)', result_text)
                    if match:
                        value = float(match.group(1))
                        confidence = value if value <= 1 else value / 100
                    
                    return {
                        'category': category,
                        'confidence': confidence,
                        'success': True
                    }
                else:
                    print(f"❌ API错误 {response.status_code}")
                    
            except requests.exceptions.Timeout:
                print(f"⏱️ 超时,重试 {attempt+1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
            except Exception as e:
                print(f"❌ 错误: {e}")
                break
        
        return {
            'category': '检测失败',
            'confidence': 0,
            'success': False
        }
    
    def get_cache_stats(self):
        """获取缓存统计"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total * 100 if total > 0 else 0
        return {
            'total': len(self.cache),
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)


class ProductConfigDialog(ctk.CTkToplevel):
    """产品保留规则配置对话框"""
    
    def __init__(self, parent, folder_stats):
        super().__init__(parent)
        
        self.title("⚙️ 产品保留规则配置")
        self.geometry("500x650")
        self.transient(parent)
        self.grab_set()
        
        self.folder_stats = folder_stats
        self.result = None
        
        self.global_scene_var = ctk.IntVar(value=4)
        self.global_material_var = ctk.IntVar(value=2)
        self.global_size_var = ctk.IntVar(value=2)
        self.global_other_var = ctk.IntVar(value=0)
        self.global_uncategorized_var = ctk.IntVar(value=0)
        
        self.setup_ui()
        center_window(self, parent)
    
    def setup_ui(self):
        header = ctk.CTkFrame(self, height=80, fg_color=("#3b8ed0", "#1f6aa5"))
        header.pack(fill="x")
        header.pack_propagate(False)
        
        ctk.CTkLabel(
            header,
            text="📋 产品保留规则配置",
            font=("Arial", 20, "bold"),
            text_color="white"
        ).pack(pady=(15, 5))
        
        ctk.CTkLabel(
            header,
            text="统一设置所有产品的保留规则",
            font=("Arial", 11),
            text_color="white"
        ).pack()
        
        config_frame = ctk.CTkFrame(
            self,
            corner_radius=10,
            fg_color=("#f5f5f5", "#2b2b2b")
        )
        config_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(
            config_frame,
            text="🌐 全局配置",
            font=("Arial", 18, "bold")
        ).pack(pady=(20, 15))
        
        categories = [
            ('🏠 场景图', self.global_scene_var),
            ('🧵 材质图', self.global_material_var),
            ('📏 尺寸图', self.global_size_var),
            ('📦 其他', self.global_other_var),
            ('❓ 未分类', self.global_uncategorized_var)
        ]
        
        for label, var in categories:
            row_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
            row_frame.pack(fill="x", padx=40, pady=12)
            
            ctk.CTkLabel(
                row_frame,
                text=label,
                font=("Arial", 14),
                width=100,
                anchor="w"
            ).pack(side="left", padx=(0, 20))
            
            ctk.CTkLabel(
                row_frame,
                text="保留:",
                font=("Arial", 13)
            ).pack(side="left", padx=5)
            
            entry = ctk.CTkEntry(
                row_frame,
                textvariable=var,
                width=100,
                height=35,
                font=("Arial", 14),
                justify="center"
            )
            entry.pack(side="left", padx=(0, 5))
            
            ctk.CTkLabel(
                row_frame,
                text="张",
                font=("Arial", 13)
            ).pack(side="left", padx=5)
        
        btn_frame = ctk.CTkFrame(self, fg_color="transparent", height=100)
        btn_frame.pack(pady=20, fill="x", padx=20)
        btn_frame.pack_propagate(False)
        
        row1 = ctk.CTkFrame(btn_frame, fg_color="transparent")
        row1.pack(pady=5)
        
        ctk.CTkButton(
            row1,
            text="✅ 应用规则",
            command=self.apply_rules,
            width=220,
            height=40,
            font=("Arial", 13, "bold"),
            fg_color="#4caf50",
            hover_color="#388e3c"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            row1,
            text="❌ 取消",
            command=self.destroy,
            width=220,
            height=40,
            font=("Arial", 13, "bold"),
            fg_color="gray",
            hover_color="#616161"
        ).pack(side="left", padx=5)
        
        row2 = ctk.CTkFrame(btn_frame, fg_color="transparent")
        row2.pack(pady=5)
        
        ctk.CTkButton(
            row2,
            text="💾 保存配置",
            command=self.save_config,
            width=220,
            height=40,
            font=("Arial", 13, "bold"),
            fg_color="#2196f3",
            hover_color="#1976d2"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            row2,
            text="📂 加载配置",
            command=self.load_config,
            width=220,
            height=40,
            font=("Arial", 13, "bold"),
            fg_color="#ff9800",
            hover_color="#f57c00"
        ).pack(side="left", padx=5)
    
    def apply_rules(self):
        """应用全局规则到所有产品"""
        try:
            rules = {
                '场景图': self.global_scene_var.get(),
                '材质图': self.global_material_var.get(),
                '尺寸图': self.global_size_var.get(),
                '其他': self.global_other_var.get(),
                '未分类': self.global_uncategorized_var.get()
            }
            
            for category, value in rules.items():
                if value < 0:
                    messagebox.showerror("错误", f"{category}的保留数量不能为负数!")
                    return
            
            result = {}
            for folder_path in self.folder_stats.keys():
                result[folder_path] = rules.copy()
            
            self.result = result
            
            total_products = len(self.folder_stats)
            messagebox.showinfo(
                "应用成功",
                f"✅ 已将规则应用到所有 {total_products} 个产品!\n\n"
                f"规则详情:\n"
                f"🏠 场景图: {rules['场景图']}张\n"
                f"🧵 材质图: {rules['材质图']}张\n"
                f"📏 尺寸图: {rules['尺寸图']}张\n"
                f"📦 其他: {rules['其他']}张\n"
                f"❓ 未分类: {rules['未分类']}张"
            )
            
            self.destroy()
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字!")
    
    def save_config(self):
        """保存配置到JSON文件"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON配置文件", "*.json")],
            title="保存配置"
        )
        
        if not filepath:
            return
        
        try:
            config = {
                'global': {
                    '场景图': self.global_scene_var.get(),
                    '材质图': self.global_material_var.get(),
                    '尺寸图': self.global_size_var.get(),
                    '其他': self.global_other_var.get(),
                    '未分类': self.global_uncategorized_var.get()
                },
                'version': '3.4',
                'description': '豆包图片分类管理系统 - 全局配置'
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            messagebox.showinfo("保存成功", f"✅ 配置已保存到:\n{filepath}")
            
        except Exception as e:
            messagebox.showerror("保存失败", f"❌ 保存失败: {e}")
    
    def load_config(self):
        """从JSON文件加载配置"""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON配置文件", "*.json")],
            title="加载配置"
        )
        
        if not filepath:
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if 'global' in config:
                global_config = config['global']
                self.global_scene_var.set(global_config.get('场景图', 4))
                self.global_material_var.set(global_config.get('材质图', 2))
                self.global_size_var.set(global_config.get('尺寸图', 2))
                self.global_other_var.set(global_config.get('其他', 0))
                self.global_uncategorized_var.set(global_config.get('未分类', 0))
            else:
                self.global_scene_var.set(config.get('场景图', 4))
                self.global_material_var.set(config.get('材质图', 2))
                self.global_size_var.set(config.get('尺寸图', 2))
                self.global_other_var.set(config.get('其他', 0))
                self.global_uncategorized_var.set(config.get('未分类', 0))
            
            messagebox.showinfo(
                "加载成功",
                f"✅ 配置已加载!\n\n"
                f"🏠 场景图: {self.global_scene_var.get()}张\n"
                f"🧵 材质图: {self.global_material_var.get()}张\n"
                f"📏 尺寸图: {self.global_size_var.get()}张\n"
                f"📦 其他: {self.global_other_var.get()}张\n"
                f"❓ 未分类: {self.global_uncategorized_var.get()}张"
            )
            
        except Exception as e:
            messagebox.showerror("加载失败", f"❌ 加载失败: {e}")


class ImageCard(ctk.CTkFrame):
    """图片卡片组件"""
    
    def __init__(self, master, image_data, **kwargs):
        super().__init__(master, **kwargs)
        self.image_data = image_data
        self.selected = False
        
        self.configure(
            width=160,
            height=220,
            corner_radius=10,
            fg_color=("white", "gray20"),
            border_width=2,
            border_color=("gray80", "gray40")
        )
        
        self.image_label = ctk.CTkLabel(self, text="")
        self.image_label.pack(pady=(10, 5))
        
        try:
            img = Image.open(image_data['path'])
            img.thumbnail((140, 100), Image.Resampling.LANCZOS)
            photo = ctk.CTkImage(
                light_image=img,
                dark_image=img,
                size=(140, 100)
            )
            self.image_label.configure(image=photo)
            self.image_label.image = photo
        except Exception as e:
            print(f"加载图片失败: {e}")
        
        filename = image_data['filename']
        if len(filename) > 20:
            filename = filename[:17] + '...'
        
        self.name_label = ctk.CTkLabel(
            self,
            text=filename,
            font=("Arial", 11),
            wraplength=140
        )
        self.name_label.pack(pady=5)
        
        confidence_text = f"置信度: {image_data['confidence']:.0%}"
        self.confidence_label = ctk.CTkLabel(
            self,
            text=confidence_text,
            font=("Arial", 10),
            text_color="gray"
        )
        self.confidence_label.pack(pady=5)
        
        self.checkbox = ctk.CTkCheckBox(
            self,
            text="选择",
            command=self.toggle_selection,
            width=60
        )
        self.checkbox.pack(pady=5)
    
    def toggle_selection(self):
        self.selected = self.checkbox.get()
        if self.selected:
            self.configure(border_color=("#3b8ed0", "#1f6aa5"))
        else:
            self.configure(border_color=("gray80", "gray40"))


class MainApp(ctk.CTk):
    """主应用程序 - 手动API配置版"""
    
    def __init__(self):
        super().__init__()
        
        self.title("豆包图片分类管理系统 v3.4 (手动API配置版)")
        self.geometry("1200x800")
        
        # 🔥 加载API配置
        self.api_config = self.load_api_config()
        
        # 初始化检测器(可能未配置)
        self.detector = DoubaoDetector(
            api_key=self.api_config.get('api_key'),
            model_id=self.api_config.get('model_id')
        )
        
        self.root_folder = None
        self.products_data = {}
        self.product_rules = {}
        self.image_cards = {}
        self.is_processing = False
        
        self.setup_ui()
        
        # 🔥 首次启动检查API配置
        if not self.detector.is_configured():
            self.after(500, self.show_api_config_prompt)
    
    def load_api_config(self):
        """加载API配置"""
        config_file = "api_config.json"
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"配置加载失败: {e}")
        return {}
    
    def save_api_config(self, config):
        """保存API配置"""
        config_file = "api_config.json"
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"配置保存失败: {e}")
            return False
    
    def show_api_config_prompt(self):
        """首次启动提示配置API"""
        result = messagebox.askyesno(
            "欢迎使用",
            "👋 欢迎使用豆包图片分类管理系统!\n\n"
            "检测到尚未配置API密钥\n\n"
            "是否现在配置?"
        )
        if result:
            self.open_api_config()
    
    def open_api_config(self):
        """打开API配置对话框"""
        dialog = APIConfigDialog(self, self.api_config)
        self.wait_window(dialog)
        
        if dialog.result:
            # 保存配置
            self.api_config = dialog.result
            self.save_api_config(self.api_config)
            
            # 更新检测器
            self.detector.update_config(
                self.api_config['api_key'],
                self.api_config['model_id']
            )
            
            # 更新状态
            self.update_api_status()
    
    def update_api_status(self):
        """更新API状态显示"""
        if self.detector.is_configured():
            self.btn_api_config.configure(
                text=f"🔧 API配置 ✓",
                fg_color="#4caf50",
                hover_color="#388e3c"
            )
        else:
            self.btn_api_config.configure(
                text="🔧 API配置 ⚠️",
                fg_color="#ff9800",
                hover_color="#f57c00"
            )
    
    def setup_ui(self):
        # 标题区域
        header = ctk.CTkFrame(self, height=100, fg_color=("#3b8ed0", "#1f6aa5"))
        header.pack(fill="x")
        header.pack_propagate(False)
        
        ctk.CTkLabel(
            header,
            text="🎨 豆包图片分类管理系统",
            font=("Arial", 24, "bold"),
            text_color="white"
        ).pack(pady=(20, 5))
        
        ctk.CTkLabel(
            header,
            text="多线程并发 · 智能缓存 · 手动API配置 | v3.4",
            font=("Arial", 12),
            text_color="white"
        ).pack()
        
        # 工具栏
        toolbar = ctk.CTkFrame(self, height=60)
        toolbar.pack(fill="x", padx=10, pady=10)
        toolbar.pack_propagate(False)
        
        # 🔥 API配置按钮(左侧第一个)
        self.btn_api_config = ctk.CTkButton(
            toolbar,
            text="🔧 API配置",
            command=self.open_api_config,
            width=140,
            height=40,
            fg_color="#ff9800",
            hover_color="#f57c00"
        )
        self.btn_api_config.pack(side="left", padx=5)
        
        self.btn_select = ctk.CTkButton(
            toolbar,
            text="📂 选择根文件夹",
            command=self.select_root_folder,
            width=140,
            height=40
        )
        self.btn_select.pack(side="left", padx=5)
        
        self.btn_detect = ctk.CTkButton(
            toolbar,
            text="🔍 批量检测",
            command=self.start_batch_detection,
            width=140,
            height=40,
            state="disabled"
        )
        self.btn_detect.pack(side="left", padx=5)
        
        self.btn_config = ctk.CTkButton(
            toolbar,
            text="⚙️ 产品配置",
            command=self.open_product_config,
            width=140,
            height=40,
            fg_color="#9c27b0",
            hover_color="#7b1fa2"
        )
        self.btn_config.pack(side="left", padx=5)
        
        self.btn_apply = ctk.CTkButton(
            toolbar,
            text="✨ 应用清理",
            command=self.apply_cleanup,
            width=140,
            height=40,
            fg_color="#4caf50",
            hover_color="#388e3c"
        )
        self.btn_apply.pack(side="left", padx=5)
        
        self.btn_export = ctk.CTkButton(
            toolbar,
            text="📦 导出图片",
            command=self.export_images_by_category,
            width=140,
            height=40
        )
        self.btn_export.pack(side="left", padx=5)
        
        self.btn_cache = ctk.CTkButton(
            toolbar,
            text="📊 缓存统计",
            command=self.show_cache_stats,
            width=140,
            height=40,
            fg_color="#607d8b",
            hover_color="#455a64"
        )
        self.btn_cache.pack(side="left", padx=5)
        
        self.status_label = ctk.CTkLabel(
            toolbar,
            text="👋 就绪",
            font=("Arial", 12)
        )
        self.status_label.pack(side="right", padx=20)
        
        # 进度条
        self.progress = ctk.CTkProgressBar(self, height=3)
        self.progress.pack(fill="x", padx=10)
        self.progress.set(0)
        self.progress.pack_forget()
        
        # 主内容区域
        self.scroll_frame = ctk.CTkScrollableFrame(
            self,
            fg_color=("gray95", "gray10")
        )
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.show_welcome()
        
        # 🔥 更新API状态显示
        self.update_api_status()
    
    def show_welcome(self):
        welcome_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        welcome_frame.pack(expand=True)
        
        ctk.CTkLabel(
            welcome_frame,
            text="👆 请先配置API,然后选择产品文件夹",
            font=("Arial", 18),
            text_color="gray"
        ).pack(pady=50)
        
        if not self.detector.is_configured():
            ctk.CTkButton(
                welcome_frame,
                text="🔧 立即配置API",
                command=self.open_api_config,
                width=200,
                height=50,
                font=("Arial", 16, "bold"),
                fg_color="#ff9800",
                hover_color="#f57c00"
            ).pack(pady=20)
    
    def clear_content(self):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
    
    def select_root_folder(self):
        folder = filedialog.askdirectory(title="选择包含多个产品文件夹的根目录")
        if folder:
            self.root_folder = folder
            self.btn_detect.configure(state="normal")
            product_folders = self.scan_product_folders(folder)
            text = f"📁 找到 {len(product_folders)} 个产品文件夹"
            self.status_label.configure(text=text)
    
    def scan_product_folders(self, root_folder):
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        product_folders = []
        try:
            for item in os.listdir(root_folder):
                item_path = os.path.join(root_folder, item)
                if os.path.isdir(item_path):
                    files = os.listdir(item_path)
                    has_images = any(
                        Path(f).suffix.lower() in image_extensions
                        for f in files
                    )
                    if has_images:
                        product_folders.append(item_path)
        except Exception as e:
            print(f"扫描错误: {e}")
        return product_folders
    
    def start_batch_detection(self):
        # 🔥 检查API配置
        if not self.detector.is_configured():
            result = messagebox.askyesno(
                "API未配置",
                "❌ 尚未配置API密钥!\n\n是否现在配置?"
            )
            if result:
                self.open_api_config()
            return
        
        if not self.root_folder:
            return
            
        self.btn_detect.configure(state="disabled")
        self.status_label.configure(text="🔄 批量检测中...")
        self.progress.pack(fill="x", padx=10)
        self.progress.set(0)
        
        thread = threading.Thread(target=self.batch_detection_worker, daemon=True)
        thread.start()
    
    def batch_detection_worker(self):
        """优化版批量检测"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        product_folders = self.scan_product_folders(self.root_folder)
        self.products_data.clear()
        
        all_tasks = []
        for folder_path in product_folders:
            try:
                files = os.listdir(folder_path)
                image_files = [
                    (folder_path, os.path.join(folder_path, f))
                    for f in files
                    if Path(f).suffix.lower() in image_extensions
                ]
                all_tasks.extend(image_files)
            except Exception as e:
                print(f"扫描失败 {folder_path}: {e}")
        
        total_images = len(all_tasks)
        if total_images == 0:
            self.after(0, lambda: messagebox.showwarning("提示", "未找到图片!"))
            self.after(0, self.detection_finished)
            return
        
        processed = 0
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_task = {
                executor.submit(self.detector.detect_image, img_path): (folder_path, img_path)
                for folder_path, img_path in all_tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_task):
                folder_path, img_path = future_to_task[future]
                processed += 1
                
                progress = processed / total_images
                self.after(0, lambda p=progress: self.progress.set(p))
                
                elapsed = time.time() - start_time
                speed = processed / elapsed if elapsed > 0 else 0
                eta = (total_images - processed) / speed if speed > 0 else 0
                
                stats = self.detector.get_cache_stats()
                
                status_text = (
                    f"🔄 {processed}/{total_images} | "
                    f"速度:{speed:.1f}张/秒 | "
                    f"剩余:{eta/60:.1f}分钟 | "
                    f"缓存命中:{stats['hit_rate']:.0f}%"
                )
                self.after(0, lambda t=status_text: self.status_label.configure(text=t))
                
                try:
                    result = future.result()
                    category = result['category']
                    
                    if folder_path not in self.products_data:
                        self.products_data[folder_path] = {}
                    if category not in self.products_data[folder_path]:
                        self.products_data[folder_path][category] = []
                    
                    self.products_data[folder_path][category].append({
                        'path': img_path,
                        'filename': os.path.basename(img_path),
                        'confidence': result['confidence']
                    })
                    
                except Exception as e:
                    print(f"处理失败 {img_path}: {e}")
        
        self.detector.save_cache()
        
        total_time = time.time() - start_time
        stats = self.detector.get_cache_stats()
        
        print(f"\n{'='*50}")
        print(f"✅ 检测完成!")
        print(f"总图片数: {total_images}")
        print(f"总耗时: {total_time/60:.1f} 分钟")
        print(f"平均速度: {total_images/total_time:.1f} 张/秒")
        print(f"缓存命中: {stats['hits']}/{total_images} ({stats['hit_rate']:.1f}%)")
        print(f"{'='*50}\n")
        
        self.after(0, self.detection_finished)
    
    def detection_finished(self):
        self.progress.pack_forget()
        self.btn_detect.configure(state="normal")
        total_images = sum(
            len(images)
            for product in self.products_data.values()
            for images in product.values()
        )
        status_text = f"✅ 完成! {len(self.products_data)}个产品,共{total_images}张图片"
        self.status_label.configure(text=status_text)
        self.refresh_display()
    
    def show_cache_stats(self):
        """显示缓存统计"""
        stats = self.detector.get_cache_stats()
        info = f"""
📊 缓存统计信息

总缓存记录: {stats['total']} 条
本次命中: {stats['hits']} 次
本次未命中: {stats['misses']} 次
命中率: {stats['hit_rate']:.1f}%

💡 缓存说明:
• 相同图片只检测一次
• 缓存保存在本地,重启后依然有效
• 如需重新检测,可清空缓存
        """
        
        dialog = ctk.CTkToplevel(self)
        dialog.title("📊 缓存统计")
        dialog.geometry("400x350")
        dialog.transient(self)
        
        ctk.CTkLabel(
            dialog,
            text=info.strip(),
            font=("Arial", 13),
            justify="left"
        ).pack(pady=20, padx=20)
        
        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=10)
        
        ctk.CTkButton(
            btn_frame,
            text="🗑️ 清空缓存",
            command=lambda: self.clear_cache_confirm(dialog),
            width=140,
            height=40,
            fg_color="#d32f2f",
            hover_color="#b71c1c"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame,
            text="✅ 关闭",
            command=dialog.destroy,
            width=140,
            height=40
        ).pack(side="left", padx=5)
        
        center_window(dialog, self)
    
    def clear_cache_confirm(self, parent_dialog):
        """确认清空缓存"""
        if messagebox.askyesno("确认", "确定清空所有缓存吗?\n\n清空后下次检测将重新调用API"):
            self.detector.clear_cache()
            parent_dialog.destroy()
            messagebox.showinfo("完成", "✅ 缓存已清空!")
    
    def open_product_config(self):
        if not self.products_data:
            messagebox.showwarning("警告", "请先完成批量检测!")
            return
        
        folder_stats = {
            fp: {cat: len(imgs) for cat, imgs in cats.items()}
            for fp, cats in self.products_data.items()
        }
        
        dialog = ProductConfigDialog(self, folder_stats)
        self.wait_window(dialog)
        
        if dialog.result:
            self.product_rules = dialog.result
            self.refresh_display()
            messagebox.showinfo("成功", "✅ 保留规则已设置!")
    
    def apply_cleanup(self):
        """应用清理规则（后台线程版）"""
        if self.is_processing:
            messagebox.showwarning("警告", "正在处理中,请稍候...")
            return
            
        if not self.product_rules:
            messagebox.showwarning("警告", "请先设置产品配置!")
            return
        
        to_delete_count = sum(
            max(0, len(self.products_data.get(fp, {}).get(cat, [])) - keep)
            for fp, rules in self.product_rules.items()
            for cat, keep in rules.items()
        )
        
        if to_delete_count == 0:
            messagebox.showinfo("提示", "没有需要清理的图片!")
            return
        
        confirm_text = f"即将删除 {to_delete_count} 张多余图片\n\n此操作不可恢复,确定继续吗?"
        if not messagebox.askyesno("确认清理", confirm_text):
            return
        
        self.is_processing = True
        self.btn_apply.configure(state="disabled", text="🔄 清理中...")
        self.progress.pack(fill="x", padx=10)
        self.progress.set(0)
        
        thread = threading.Thread(
            target=self.cleanup_worker,
            args=(to_delete_count,),
            daemon=True
        )
        thread.start()
    
    def cleanup_worker(self, total_to_delete):
        """清理工作线程"""
        deleted = 0
        processed = 0
        
        total_tasks = sum(
            len(self.products_data.get(fp, {}).get(cat, []))
            for fp, rules in self.product_rules.items()
            for cat in rules.keys()
        )
        
        for folder_path, rules in self.product_rules.items():
            for category, keep_count in rules.items():
                if category in self.products_data.get(folder_path, {}):
                    images = self.products_data[folder_path][category]
                    images.sort(key=lambda x: x['confidence'], reverse=True)
                    
                    for img in images[keep_count:]:
                        try:
                            os.remove(img['path'])
                            deleted += 1
                            
                            processed += 1
                            progress = processed / total_tasks
                            self.after(0, lambda p=progress: self.progress.set(p))
                            
                            status = f"🗑️ 清理中: {deleted}/{total_to_delete}"
                            self.after(0, lambda s=status: self.status_label.configure(text=s))
                            
                        except Exception as e:
                            print(f"删除失败: {e}")
                    
                    self.products_data[folder_path][category] = images[:keep_count]
        
        self.after(0, lambda: self.cleanup_finished(deleted))
    
    def cleanup_finished(self, deleted_count):
        """清理完成回调"""
        self.is_processing = False
        self.progress.pack_forget()
        self.btn_apply.configure(state="normal", text="✨ 应用清理")
        self.status_label.configure(text=f"✅ 已删除 {deleted_count} 张图片")
        self.refresh_display()
        messagebox.showinfo("完成", f"✅ 已删除 {deleted_count} 张图片!")
    
    def export_images_by_category(self):
        """按类别导出图片"""
        if not self.products_data:
            messagebox.showwarning("警告", "暂无数据!")
            return
        
        dialog = ctk.CTkToplevel(self)
        dialog.title("📦 选择导出类别")
        dialog.geometry("400x450")
        dialog.transient(self)
        dialog.grab_set()
        
        ctk.CTkLabel(
            dialog,
            text="请选择要导出的图片类别",
            font=("Arial", 16, "bold")
        ).pack(pady=20)
        
        category_counts = {
            '场景图': 0,
            '材质图': 0,
            '尺寸图': 0,
            '其他': 0,
            '未分类': 0
        }
        
        for product_data in self.products_data.values():
            for category, images in product_data.items():
                if category in category_counts:
                    category_counts[category] += len(images)
        
        options_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        options_frame.pack(pady=10, padx=30, fill="both", expand=True)
        
        selected_category = ctk.StringVar(value="场景图")
        
        icon_map = {
            '场景图': '🏠',
            '材质图': '🧵',
            '尺寸图': '📏',
            '其他': '📦',
            '未分类': '❓'
        }
        
        for category in ['场景图', '材质图', '尺寸图', '其他', '未分类']:
            count = category_counts[category]
            if count == 0:
                continue
                
            radio_text = f"{icon_map[category]} {category} ({count}张)"
            
            ctk.CTkRadioButton(
                options_frame,
                text=radio_text,
                variable=selected_category,
                value=category,
                font=("Arial", 14),
                radiobutton_width=20,
                radiobutton_height=20
            ).pack(anchor="w", pady=10, padx=20)
        
        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=20)
        
        def do_export():
            category = selected_category.get()
            dialog.destroy()
            self.execute_category_export(category)
        
        ctk.CTkButton(
            btn_frame,
            text="✅ 开始导出",
            command=do_export,
            width=140,
            height=40,
            font=("Arial", 14, "bold"),
            fg_color="#4caf50",
            hover_color="#388e3c"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame,
            text="❌ 取消",
            command=dialog.destroy,
            width=140,
            height=40,
            font=("Arial", 14, "bold"),
            fg_color="gray"
        ).pack(side="left", padx=5)
        
        center_window(dialog, self)
    
    def execute_category_export(self, category):
        """执行按类别导出"""
        output_folder = filedialog.askdirectory(title=f"选择导出文件夹 - {category}")
        
        if not output_folder:
            return
        
        category_folder = os.path.join(output_folder, category)
        os.makedirs(category_folder, exist_ok=True)
        
        exported_count = 0
        failed_count = 0
        
        for folder_path, categories in self.products_data.items():
            if category in categories:
                product_name = os.path.basename(folder_path)
                product_output = os.path.join(category_folder, product_name)
                os.makedirs(product_output, exist_ok=True)
                
                for img_data in categories[category]:
                    try:
                        src_path = img_data['path']
                        filename = img_data['filename']
                        dst_path = os.path.join(product_output, filename)
                        
                        shutil.copy2(src_path, dst_path)
                        exported_count += 1
                    except Exception as e:
                        print(f"导出失败: {e}")
                        failed_count += 1
        
        result_text = f"✅ 导出完成!\n\n"
        result_text += f"类别: {category}\n"
        result_text += f"成功: {exported_count}张\n"
        if failed_count > 0:
            result_text += f"失败: {failed_count}张\n"
        result_text += f"\n保存位置:\n{category_folder}"
        
        messagebox.showinfo("导出完成", result_text)
    
    def refresh_display(self):
        self.clear_content()
        self.image_cards.clear()
        
        if not self.products_data:
            self.show_welcome()
            return
        
        for folder_path in sorted(self.products_data.keys()):
            self.create_product_section(folder_path)
    
    def create_product_section(self, folder_path):
        product_name = os.path.basename(folder_path)
        
        section = ctk.CTkFrame(self.scroll_frame, corner_radius=10)
        section.pack(fill="x", pady=10, padx=5)
        
        header = ctk.CTkFrame(section, fg_color=("#e3f2fd", "#1e3a5f"))
        header.pack(fill="x", padx=10, pady=10)
        
        title_text = f"📦 {product_name}"
        if folder_path in self.product_rules:
            rules_list = [
                f"{cat}:{count}张"
                for cat, count in self.product_rules[folder_path].items()
                if count > 0
            ]
            rules_text = " | ".join(rules_list)
            if rules_text:
                title_text += f" (规则: {rules_text})"
        
        ctk.CTkLabel(
            header,
            text=title_text,
            font=("Arial", 16, "bold"),
            anchor="w"
        ).pack(side="left", padx=10, pady=8)
        
        categories = [
            '场景图', '材质图', '尺寸图',
            '其他', '未分类', '检测失败'
        ]
        
        for category in categories:
            if category in self.products_data[folder_path]:
                self.create_category_subsection(section, folder_path, category)
    
    def create_category_subsection(self, parent, folder_path, category):
        images = self.products_data[folder_path][category]
        if not images:
            return
        
        cat_header = ctk.CTkFrame(parent, fg_color="transparent")
        cat_header.pack(fill="x", padx=20, pady=(10, 5))
        
        icon_map = {
            '材质图': '🧵',
            '尺寸图': '📏',
            '场景图': '🏠',
            '其他': '📦',
            '未分类': '❓',
            '检测失败': '⚠️'
        }
        
        keep_count = self.product_rules.get(folder_path, {}).get(category, None)
        
        title_text = f"{icon_map.get(category, '📷')} {category} ({len(images)}张"
        if keep_count is not None:
            excess = max(0, len(images) - keep_count)
            title_text += f" | 保留{keep_count}"
            if excess > 0:
                title_text += f" | ⚠️多余{excess}"
        title_text += ")"
        
        ctk.CTkLabel(
            cat_header,
            text=title_text,
            font=("Arial", 14, "bold")
        ).pack(side="left")
        
        btn_frame = ctk.CTkFrame(cat_header, fg_color="transparent")
        btn_frame.pack(side="right")
        
        key = (folder_path, category)
        
        ctk.CTkButton(
            btn_frame,
            text="✓ 全选",
            command=lambda k=key: self.select_all_in_category(k),
            width=70,
            height=28
        ).pack(side="left", padx=3)
        
        ctk.CTkButton(
            btn_frame,
            text="○ 取消",
            command=lambda k=key: self.deselect_all_in_category(k),
            width=70,
            height=28
        ).pack(side="left", padx=3)
        
        ctk.CTkButton(
            btn_frame,
            text="🗑️ 删除选中",
            command=lambda k=key: self.delete_selected_in_category(k),
            width=90,
            height=28,
            fg_color="#d32f2f",
            hover_color="#b71c1c"
        ).pack(side="left", padx=3)
        
        grid_frame = ctk.CTkFrame(parent, fg_color="transparent")
        grid_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        images_sorted = sorted(
            images,
            key=lambda x: x['confidence'],
            reverse=True
        )
        
        self.image_cards[key] = []
        
        row_frame = None
        for idx, img_data in enumerate(images_sorted):
            if idx % 6 == 0:
                row_frame = ctk.CTkFrame(grid_frame, fg_color="transparent")
                row_frame.pack(fill="x", pady=5)
            
            card = ImageCard(row_frame, img_data)
            card.pack(side="left", padx=5)
            self.image_cards[key].append(card)
    
    def select_all_in_category(self, key):
        if key in self.image_cards:
            for card in self.image_cards[key]:
                card.checkbox.select()
                card.toggle_selection()
    
    def deselect_all_in_category(self, key):
        if key in self.image_cards:
            for card in self.image_cards[key]:
                card.checkbox.deselect()
                card.toggle_selection()
    
    def delete_selected_in_category(self, key):
        if key not in self.image_cards:
            return
        
        folder_path, category = key
        selected = [card for card in self.image_cards[key] if card.selected]
        
        if not selected:
            messagebox.showwarning("警告", "请先选择要删除的图片!")
            return
        
        confirm_text = f"确定删除 {len(selected)} 张图片吗?\n此操作不可恢复!"
        if not messagebox.askyesno("确认删除", confirm_text):
            return
        
        deleted = 0
        for card in selected:
            try:
                os.remove(card.image_data['path'])
                self.products_data[folder_path][category].remove(card.image_data)
                deleted += 1
            except Exception as e:
                print(f"删除失败: {e}")
        
        self.refresh_display()
        messagebox.showinfo("完成", f"已删除 {deleted} 张图片")


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()


