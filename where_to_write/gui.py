"""Tk desktop microscope. All Qwen work runs outside Tk's UI thread."""
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .memory import COLORS, FactMemory
from .probe import PROMPTS, QwenProbe

DISPLAY = {
    'baseline': 'No memory',
    'subject': 'Memory → subject token',
    'answer': 'Memory → answer token',
    'other_token': 'Memory → word “color”',
    'random_subject': 'Random pulse → subject',
    'wrong_subject': 'Wrong color → subject',
}


class Window:
    def __init__(self, root):
        self.root = root
        root.title('WhereToWrite — memory-site microscope')
        root.geometry('1060x740')
        root.minsize(840, 620)
        self.facts = FactMemory()
        self.model = None
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.running = False
        self.name = tk.StringVar(value='Vera')
        self.color = tk.StringVar(value='yellow')
        self.gain = tk.DoubleVar(value=0.5)
        self.layer = tk.IntVar(value=16)
        self.model_name = tk.StringVar(value='Qwen/Qwen2.5-0.5B-Instruct')
        self.template = tk.StringVar(value=PROMPTS[0])
        self.status = tk.StringVar(value='Step 1: Load Qwen. No fact is sent in the question prompt.')
        self._build()
        self.facts.observe('Vera', 'yellow')
        self.facts.observe('Eero', 'blue')
        self._refresh()
        root.protocol('WM_DELETE_WINDOW', self._close)

    def _build(self):
        root = self.root
        root.configure(padx=13, pady=9)
        heading = ttk.Frame(root)
        heading.pack(fill='x', pady=(0, 8))
        ttk.Label(heading, text='WHERE TO WRITE', font=('Segoe UI', 18, 'bold')).pack(side='left')
        ttk.Label(heading, text='  read a fact → change a token → watch Qwen', font=('Segoe UI', 10)).pack(side='left')
        load = ttk.LabelFrame(root, text='Frozen model (one download; one load per session)', padding=8)
        load.pack(fill='x')
        ttk.Entry(load, textvariable=self.model_name, width=41).pack(side='left', padx=3)
        ttk.Label(load, text='Layer').pack(side='left', padx=(12, 2))
        ttk.Spinbox(load, from_=0, to=23, width=4, textvariable=self.layer).pack(side='left')
        self.load_button = ttk.Button(load, text='Load Qwen', command=self._load)
        self.load_button.pack(side='left', padx=8)
        ttk.Label(load, text='Gain').pack(side='left', padx=(8, 3))
        ttk.Scale(load, variable=self.gain, from_=0.0, to=8.0, length=155).pack(side='left')
        ttk.Label(load, textvariable=self.gain, width=6).pack(side='left')

        editor = ttk.LabelFrame(root, text='Experience — explicit memory (oracle name binding)', padding=8)
        editor.pack(fill='x', pady=8)
        ttk.Label(editor, text='Name').pack(side='left')
        ttk.Entry(editor, textvariable=self.name, width=16).pack(side='left', padx=(5, 12))
        ttk.Label(editor, text='Observed color').pack(side='left')
        ttk.Combobox(editor, textvariable=self.color, values=COLORS, state='readonly', width=10).pack(side='left', padx=6)
        ttk.Button(editor, text='WRITE / rewrite', command=self._write).pack(side='left', padx=8)
        ttk.Button(editor, text='Save facts', command=self._save).pack(side='right')
        ttk.Button(editor, text='Load facts', command=self._open).pack(side='right', padx=5)

        row = ttk.Frame(root)
        row.pack(fill='both', expand=True)
        left = ttk.LabelFrame(row, text='Persistent facts (not in Qwen context)', padding=7)
        left.pack(side='left', fill='y', padx=(0, 8))
        self.memory_list = tk.Listbox(left, width=26, height=18, exportselection=False, font=('Consolas', 10))
        self.memory_list.pack(fill='both', expand=True)
        self.memory_list.bind('<<ListboxSelect>>', self._select)
        self.names = []
        right = ttk.Frame(row)
        right.pack(side='left', fill='both', expand=True)
        ask = ttk.LabelFrame(right, text='Probe: where should the value enter?', padding=7)
        ask.pack(fill='x')
        ttk.Combobox(ask, textvariable=self.template, state='readonly', values=PROMPTS, width=54).pack(side='left', fill='x', expand=True, padx=(0, 7))
        self.ask_button = ttk.Button(ask, text='ASK + compare', command=self._ask, state='disabled')
        self.ask_button.pack(side='right')
        self.output = tk.Text(right, height=5, font=('Consolas', 10), wrap='word', state='disabled', background='#121820', foreground='#e8eef7')
        self.output.pack(fill='x', pady=(8, 5))
        self.bars = tk.Canvas(right, background='#111827', height=280, highlightthickness=0)
        self.bars.pack(fill='both', expand=True)
        ttk.Label(root, textvariable=self.status, anchor='w').pack(fill='x', pady=(8, 1))
        ttk.Label(root, text='Research boundary: explicit dictionary picks WHO and WHAT. Qwen is frozen. Only intervention LOCATION is tested.',
                  foreground='#926d2c').pack(fill='x')

    def _refresh(self):
        self.memory_list.delete(0, 'end')
        self.names = list(self.facts.facts.keys())
        for n in self.names:
            self.memory_list.insert('end', f'{n:<16} → {self.facts.facts[n]}')

    def _select(self, _event):
        selection = self.memory_list.curselection()
        if selection:
            self.name.set(self.names[selection[0]])
            self.color.set(self.facts.get(self.name.get()))

    def _write(self):
        try:
            previous = self.facts.observe(self.name.get(), self.color.get())
            self._refresh()
            self.status.set(f'Written: {self.name.get()} = {self.color.get()} (was {previous or "unknown"}).')
        except ValueError as exc:
            messagebox.showerror('Invalid fact', str(exc))

    def _save(self):
        f = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON', '*.json')])
        if f:
            self.facts.save(f)

    def _open(self):
        f = filedialog.askopenfilename(filetypes=[('JSON', '*.json')])
        if f:
            try:
                self.facts.load(f)
                self._refresh()
            except (ValueError, OSError) as exc:
                messagebox.showerror('Invalid facts file', str(exc))

    def _future(self, fn, callback):
        self.running = True
        self.load_button.configure(state='disabled')
        self.ask_button.configure(state='disabled')
        future = self.pool.submit(fn)
        def poll():
            if not future.done():
                self.root.after(100, poll)
                return
            self.running = False
            try:
                callback(future.result())
            except Exception as exc:
                self.status.set(f'Error: {exc}')
                messagebox.showerror('Qwen error', str(exc))
            finally:
                # The successful load callback assigns self.model. Enable Ask
                # AFTER that callback, not before it, or it stays disabled.
                self.load_button.configure(state='normal')
                self.ask_button.configure(state='normal' if self.model else 'disabled')
        self.root.after(100, poll)

    def _load(self):
        name, layer = self.model_name.get().strip(), int(self.layer.get())
        self.status.set(f'Loading {name}... GPU recommended; first download may take time.')
        def finished(model):
            self.model = model
            self.status.set(f'Loaded {name}, layer {layer}, using {model.device}. Ask about a stored fact.')
        self._future(lambda: QwenProbe(name, layer=layer), finished)

    def _ask(self):
        if not self.model:
            return
        name = self.name.get().strip()
        value = self.facts.get(name)
        if value is None:
            messagebox.showinfo('No fact', 'WRITE an observation for this name first.')
            return
        gain = float(self.gain.get())
        template = self.template.get()
        self.status.set(f'Comparing six interventions on {name}; Qwen weights remain unchanged...')
        self._future(lambda: self.model.evaluate(name, value, template, gain),
                     lambda result: self._draw(result, name, value, gain))

    def _draw(self, scores, name, color, gain):
        self.output.configure(state='normal')
        self.output.delete('1.0', 'end')
        self.output.insert('end', f'QUESTION   {self.template.get().format(name=name)}\n')
        self.output.insert('end', f'FACT       {name} = {color}   |  layer={self.model.layer}, gain={gain:.2f}\n')
        self.output.insert('end', 'Four-color probabilities below are CONDITIONAL on choosing one of four colors.\n')
        self.output.insert('end', 'Exact strict = lowercase top color token; capitalized color tracked separately.\n')
        self.output.configure(state='disabled')
        c = self.bars
        c.delete('all')
        c.update_idletasks()
        width = max(c.winfo_width(), 550)
        x0, x1 = 204, width - 180
        rows = list(scores.items())
        for i, (arm, res) in enumerate(rows):
            y = 20 + i * 42
            prob = res.conditional_color_probs[color]
            fill = '#35c594' if res.semantic_top_correct else '#edab59' if res.choice_correct else '#d86469'
            c.create_text(10, y, text=DISPLAY[arm], fill='#e9eff5', anchor='w', font=('Segoe UI', 9))
            c.create_rectangle(x0, y-8, x1, y+8, fill='#273348', outline='')
            c.create_rectangle(x0, y-8, x0 + (x1-x0)*prob, y+8, fill=fill, outline='')
            c.create_text(x1 + 8, y, text=f'{prob:5.1%}  → {res.choice}', fill='#e8eef7', anchor='w', font=('Consolas', 9))
            c.create_text(x1+8, y+14, text=('exact strict ✓' if res.strict_correct else 'case variant ✓' if res.semantic_top_correct else f'top={res.top_token!r}'),
                          fill='#aebcd0', anchor='w', font=('Consolas', 8))
        self.status.set('Finished. Green = correct top-word color (case-insensitive); orange = correct only among four lowercase tokens.')

    def _close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use('clam')
    except tk.TclError:
        pass
    Window(root)
    root.mainloop()
