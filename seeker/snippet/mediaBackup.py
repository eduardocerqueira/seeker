#date: 2022-03-04T17:01:17Z
#url: https://api.github.com/gists/33e98fd44a23a46e5e83ada0000bb3cb
#owner: https://api.github.com/users/JonasBernard

import shutil
import datetime
import threading

import gi
import os

gi.require_version("Gtk", "3.0")
from gi.repository import Gio, Gtk


class ProcessDialog(Gtk.Dialog):
    def __init__(self, parent):
        Gtk.Dialog.__init__(self, title="Backup im process", transient_for=parent, flags=0)
        self.add_buttons(
            Gtk.STOCK_CLOSE, Gtk.ResponseType.CANCEL,
        )

        self.set_default_size(500, 300)

        self.scoll = Gtk.ScrolledWindow(hexpand=True, vexpand=True)
        self.log = Gtk.ListBox()
        self.log.set_selection_mode(Gtk.SelectionMode.NONE)
        self.grid = Gtk.Grid()

        self.scoll.add(self.log)
        self.grid.add(self.scoll)

        self.add_log("Logging content will appear here...")

        box = self.get_content_area()
        box.add(self.grid)
        self.show_all()

    def add_log(self, string):
        l = Gtk.ListBoxRow()
        text = Gtk.Label(xalign=0)
        text.set_label(string)
        l.add(text)
        self.log.add(l)
        self.show_all()


class MainWindow(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Mobile Media Backup")
        self.set_border_width(10)

        self.grid = Gtk.Grid()
        self.add(self.grid)

        self.header = Gtk.HeaderBar(title="Backup all media files from a folder that are older than a certain date")
        self.grid.attach(self.header, 0, 0, 2, 1)

        self.spinner = Gtk.Spinner()

        self.start_button = Gtk.Button(label="Start")
        self.start_button.connect("clicked", self.start)
        self.header.pack_end(self.start_button)

        self.files_box = Gtk.ListBox()
        self.files_box.set_selection_mode(Gtk.SelectionMode.NONE)
        self.grid.attach(self.files_box, 0, 1, 1, 1)

        self.source = Gtk.ListBoxRow()
        self.hbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.source.add(self.hbox)

        self.source_label = Gtk.Label()
        self.source_label.set_text("Select source")
        self.source_button = Gtk.Button(label="Select Source")
        self.source_button = Gtk.FileChooserButton(action=Gtk.FileChooserAction.SELECT_FOLDER)
        self.hbox.pack_start(self.source_label, True, True, 10)
        self.hbox.pack_end(self.source_button, True, True, 10)

        self.dest = Gtk.ListBoxRow()
        self.hbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.dest.add(self.hbox)

        self.dest_label = Gtk.Label()
        self.dest_label.set_text("Select destination")
        self.dest_button = Gtk.FileChooserButton(action=Gtk.FileChooserAction.SELECT_FOLDER)
        self.hbox.pack_start(self.dest_label, True, True, 10)
        self.hbox.pack_end(self.dest_button, True, True, 10)

        self.files_box.add(self.source)
        self.files_box.add(self.dest)

        self.calendar = Gtk.Calendar(hexpand=True, vexpand=True)
        self.grid.attach(self.calendar, 1, 1, 1, 1)

    def start(self, widget):
        if not os.path.isdir(str(self.source_button.get_filename())):
            msg = Gtk.MessageDialog(transient_for=self, flags=0, message_type=Gtk.MessageType.INFO,
                                    title="No source file selected", buttons=Gtk.ButtonsType.OK,
                                    text="There isn't any valid source file selected.")
            msg.format_secondary_text(
                "You cannot start the process before selecting a source directory."
            )
            msg.connect("response", self.dialog_response)
            msg.show()
            return

        if not os.path.isdir(str(self.dest_button.get_filename())):
            msg = Gtk.MessageDialog(transient_for=self, flags=0, message_type=Gtk.MessageType.INFO,
                                    title="No destination file selected", buttons=Gtk.ButtonsType.OK,
                                    text="There isn't any valid destination file selected.")
            msg.format_secondary_text(
                "You cannot start the process before selecting a destination directory."
            )
            msg.connect("response", self.dialog_response)
            msg.show()
            return

        cal_min_date = self.calendar.get_date()
        min_date = datetime.datetime(cal_min_date[0], cal_min_date[1] + 1, cal_min_date[2])

        dialog = ProcessDialog(self)
        process = threading.Thread(target=self.copy_to_meta, args=(dialog.add_log, min_date,
                                                                   self.source_button.get_filename(),
                                                                   self.dest_button.get_filename()))

        self.spinner.start()
        self.header.pack_start(self.spinner)
        dialog.run()

        self.spinner.stop()
        dialog.destroy()

    def dialog_response(self, widget, response_id):
        widget.destroy()

    def copy_to_meta(self, log_function, min_date, source_dir, dest_dir):
        self.copy_to(log_function, min_date, source_dir, dest_dir)
        log_function("FINISH!")

    def copy_to(self, log_function, min_date, source_dir, dest_dir):
        log_function("Scanning {0}...".format(source_dir))
        for file in os.listdir(source_dir):
            source_file = os.path.join(source_dir, file)
            dest_file = os.path.join(dest_dir, file)
            if os.path.isfile(source_file):
                if datetime.datetime.fromtimestamp(os.path.getmtime(source_file)) < min_date:
                    log_function("Found file {0} and copy to {1}.".format(source_file, dest_file))
                    shutil.move(source_file, dest_file)
                else:
                    log_function("Found file {0} but its to new.".format(source_file))
            elif os.path.isdir(source_file):
                log_function("Found directory {0} and scanning with new destination {1}".format(source_file, dest_file))
                if not os.path.isdir(dest_file):
                    log_function("Creating directory {0}.".format(dest_file))
                    os.mkdir(dest_file)
                self.copy_to(log_function, min_date, source_file, dest_file)


win = MainWindow()
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()
