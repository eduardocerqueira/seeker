#date: 2025-08-20T17:03:37Z
#url: https://api.github.com/gists/dd5d035f2672dc154b9cec022a0cd5ce
#owner: https://api.github.com/users/ynkdir

# WinUI3 example for accessing UI component in subthread.
# UI component cannot be accessed in subthread.
# Use asyncio or DispatcherQueue to execute command in mainthread.

import asyncio
import threading
import time

from win32more.Microsoft.UI.Dispatching import DispatcherQueue
from win32more.Microsoft.UI.Xaml.Media import SolidColorBrush
from win32more.Windows.UI import Colors
from win32more.xaml import XamlApplication, XamlLoader


class App(XamlApplication):
    def OnLaunched(self, args):
        self._window = XamlLoader.Load(
            self,
            """<?xml version="1.0" encoding="utf-8"?>
<Window
    xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
    xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
    xmlns:d="http://schemas.microsoft.com/expression/blend/2008"
    xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
    mc:Ignorable="d"
    xmlns:controls="using:Microsoft.UI.Xaml.Controls">

    <StackPanel>
        <Button x:Name="Button1" Click="Button1_Click" Content="Run asyncio" />
        <Rectangle x:Name="Rectangle1" Width="100" Height="100" Fill="Blue" Stroke="Red" />
        <Button x:Name="Button2" Click="Button2_Click" Content="Run DispatcherQueue" />
        <Rectangle x:Name="Rectangle2" Width="100" Height="100" Fill="Yellow" Stroke="Green" />
    </StackPanel>
</Window>
""",
        )
        self._window.Activate()

    def Button1_Click(self, sender, e):
        def worker():
            for i in range(10):
                loop.call_soon_threadsafe(lambda: self.Rectangle1.put_Fill(SolidColorBrush(Colors.Blue)))
                time.sleep(0.5)
                loop.call_soon_threadsafe(lambda: self.Rectangle1.put_Fill(SolidColorBrush(Colors.Red)))
                time.sleep(0.5)

        loop = asyncio.get_running_loop()

        self._worker = threading.Thread(target=worker)
        self._worker.start()

    def Button2_Click(self, sender, e):
        def worker():
            for i in range(10):
                dispatcher.TryEnqueue(lambda: self.Rectangle2.put_Fill(SolidColorBrush(Colors.Yellow)))
                time.sleep(0.5)
                dispatcher.TryEnqueue(lambda: self.Rectangle2.put_Fill(SolidColorBrush(Colors.Green)))
                time.sleep(0.5)

        dispatcher = DispatcherQueue.GetForCurrentThread()
        # or
        # dispatcher = self._window.DispatcherQueue

        self._worker = threading.Thread(target=worker)
        self._worker.start()


XamlApplication.Start(App)