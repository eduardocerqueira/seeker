#date: 2026-01-14T17:08:58Z
#url: https://api.github.com/gists/74cd0bd37485e59ce65642bf51785e8f
#owner: https://api.github.com/users/anjandn

# poly_diff_notebook.py
# Minimal polygon diff editor for Jupyter (ipywidgets + JS canvas)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import traitlets
import ipywidgets as widgets
from IPython.display import Javascript, display


Polys = List[List[Tuple[float, float]]]


class PolyVar(traitlets.HasTraits):
    """
    A simple traitlets-backed container you can bind to the editor.

    Usage:
        A = PolyVar(value=[[(0,0),(10,0),(10,10),(0,10)]])
        B = PolyVar(value=[])
        w = poly_diff(A, B)

        # Update from Python (UI updates)
        A.value = [[(0,0),(5,0),(5,5),(0,5)]]

        # Read after editing (Python sees it)
        print(A.value)
    """
    value = traitlets.List(default_value=[]).tag(sync=False)


@widgets.register
class PolyDiffWidget(widgets.DOMWidget):
    _view_name = traitlets.Unicode("PolyDiffView").tag(sync=True)
    _view_module = traitlets.Unicode("poly_diff_widget").tag(sync=True)
    _view_module_version = traitlets.Unicode("0.0.1").tag(sync=True)

    polysA = traitlets.List(default_value=[]).tag(sync=True)
    polysB = traitlets.List(default_value=[]).tag(sync=True)
    active = traitlets.Unicode("A").tag(sync=True)
    showA = traitlets.Bool(True).tag(sync=True)
    showB = traitlets.Bool(True).tag(sync=True)
    width = traitlets.Int(560).tag(sync=True)
    height = traitlets.Int(320).tag(sync=True)


_JS = r"""
(function(){
  if (window.__poly_diff_widget_loaded__) return;
  window.__poly_diff_widget_loaded__ = true;

  function ensureRequire(){
    if (typeof require === 'undefined' || typeof define === 'undefined'){
      console.error("poly_diff_widget: requirejs not available.");
      return false;
    }
    return true;
  }
  if (!ensureRequire()) return;

  try{ require.undef('poly_diff_widget'); }catch(e){}

  define('poly_diff_widget', ["@jupyter-widgets/base"], function(widgets){
    function clamp(v,a,b){ return Math.max(a, Math.min(b, v)); }
    function deepCopy(o){ return JSON.parse(JSON.stringify(o)); }

    function toPtObj(polys){
      // polys: [ [ [x,y], ... ], ... ]  -> [ [ {x,y}, ... ], ... ]
      const out = [];
      if (!Array.isArray(polys)) return out;
      for (const ring of polys){
        if (!Array.isArray(ring)) continue;
        const pts = [];
        for (const p of ring){
          if (!Array.isArray(p) || p.length < 2) continue;
          const x = Number(p[0]), y = Number(p[1]);
          if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
          pts.push({x:x, y:y});
        }
        // drop duplicate close point
        if (pts.length >= 2){
          const a = pts[0], b = pts[pts.length-1];
          if (a.x===b.x && a.y===b.y) pts.pop();
        }
        out.push(pts);
      }
      return out;
    }

    function toModel(polys){
      const out = [];
      for (const ring of polys){
        const pts = [];
        for (const p of ring){
          pts.push([p.x, p.y]);
        }
        out.push(pts);
      }
      return out;
    }

    function fmtNum(n){
      const r = Math.round(n);
      if (Math.abs(n-r) < 1e-9) return String(r);
      return String(Number(n.toFixed(6)));
    }
    function toPython(polys){
      const parts = polys.map(ring=>{
        const pts = ring.map(p=>`(${fmtNum(p.x)}, ${fmtNum(p.y)})`).join(", ");
        return `[${pts}]`;
      }).join(",\n  ");
      return `[\n  ${parts}\n]`;
    }

    function parsePythonPolys(s){
      try{
        let t = (s||"").trim();
        if (!t) return {ok:true, polys:[]};
        t = t.replace(/^[a-zA-Z_]\w*\s*=\s*/,"");
        t = t.replaceAll("(", "[").replaceAll(")", "]");
        t = t.replace(/,\s*]/g, "]");
        const obj = JSON.parse(t);
        if (!Array.isArray(obj)) return {ok:false, err:"Top-level must be a list/array."};
        const polys = [];
        for (const ring of obj){
          if (!Array.isArray(ring)) return {ok:false, err:"Each polygon must be a list/array of points."};
          const pts = [];
          for (const pt of ring){
            if (!Array.isArray(pt) || pt.length<2) return {ok:false, err:"Each point must be (x,y) or [x,y]."};
            const x = Number(pt[0]), y = Number(pt[1]);
            if (!Number.isFinite(x) || !Number.isFinite(y)) return {ok:false, err:"Point coordinates must be numbers."};
            pts.push([x,y]);
          }
          if (pts.length>=2){
            const a=pts[0], b=pts[pts.length-1];
            if (a[0]===b[0] && a[1]===b[1]) pts.pop();
          }
          polys.push(pts);
        }
        return {ok:true, polys: polys};
      }catch(e){
        return {ok:false, err:"Parse error: " + e.message};
      }
    }

    function dist2(ax,ay,bx,by){ const dx=ax-bx, dy=ay-by; return dx*dx+dy*dy; }
    function segDist2(px,py, ax,ay, bx,by){
      const vx=bx-ax, vy=by-ay;
      const wx=px-ax, wy=py-ay;
      const c1 = vx*wx + vy*wy;
      if (c1 <= 0) return dist2(px,py,ax,ay);
      const c2 = vx*vx + vy*vy;
      if (c2 <= c1) return dist2(px,py,bx,by);
      const t = c1 / c2;
      const cx = ax + t*vx, cy = ay + t*vy;
      return dist2(px,py,cx,cy);
    }

    function keyPt(p){ return String(p.x) + "," + String(p.y); }
    function keySeg(a,b){
      const k1 = keyPt(a), k2 = keyPt(b);
      return (k1 <= k2) ? (k1 + "|" + k2) : (k2 + "|" + k1);
    }

    function computeEdgeGroups(polys){
      // returns {key -> [edgeObj,...]}, and flat edges list
      const map = Object.create(null);
      const edges = [];
      for (let pi=0; pi<polys.length; pi++){
        const ring = polys[pi];
        const n = ring.length;
        if (n < 2) continue;
        for (let i=0; i<n; i++){
          const a = ring[i];
          const b = ring[(i+1)%n];
          const k = keySeg(a,b);
          const e = {poly:pi, idx:i, a:a, b:b, key:k};
          edges.push(e);
          if (!map[k]) map[k] = [];
          map[k].push(e);
        }
      }
      // stable ordering
      for (const k in map){
        map[k].sort((u,v)=> (u.poly - v.poly) || (u.idx - v.idx));
      }
      return {map, edges};
    }

    function offsetList(k, basePx){
      // symmetric offsets, ensure none is exactly 0
      const out = [];
      const mid = (k-1)/2;
      for (let i=0; i<k; i++){
        let o = (i - mid) * basePx;
        if (Math.abs(o) < 1e-9){
          o = basePx * 0.35 * (i%2===0 ? 1 : -1);
        }
        out.push(o);
      }
      return out;
    }

    var PolyDiffView = widgets.DOMWidgetView.extend({
      render: function(){
        this.el.innerHTML = "";
        this._root = document.createElement("div");
        this._root.style.cssText = "display:flex;flex-direction:column;gap:6px;font:12px/1.25 system-ui,-apple-system,Segoe UI,Roboto,Arial;color:#e5e7eb;";

        // toolbar
        const bar = document.createElement("div");
        bar.style.cssText = "display:flex;align-items:center;gap:10px;flex-wrap:wrap;";
        this._btnA = document.createElement("button");
        this._btnB = document.createElement("button");
        this._btnA.textContent = "A";
        this._btnB.textContent = "B";
        for (const b of [this._btnA, this._btnB]){
          b.style.cssText = "padding:4px 10px;border-radius:10px;border:1px solid #334155;background:#0b1220;color:#e5e7eb;cursor:pointer;";
        }
        this._chkA = document.createElement("input");
        this._chkB = document.createElement("input");
        this._chkA.type="checkbox"; this._chkB.type="checkbox";
        const labA = document.createElement("label");
        const labB = document.createElement("label");
        labA.style.cssText = "display:flex;align-items:center;gap:6px;color:#cbd5e1;";
        labB.style.cssText = "display:flex;align-items:center;gap:6px;color:#cbd5e1;";
        labA.appendChild(this._chkA); labA.appendChild(document.createTextNode("show A"));
        labB.appendChild(this._chkB); labB.appendChild(document.createTextNode("show B"));

        this._status = document.createElement("span");
        this._status.style.cssText = "margin-left:auto;color:#94a3b8;";

        bar.appendChild(this._btnA);
        bar.appendChild(this._btnB);
        bar.appendChild(labA);
        bar.appendChild(labB);
        bar.appendChild(this._status);

        // canvas
        this._canvas = document.createElement("canvas");
        this._canvas.style.cssText = "border:1px solid #1f2937;border-radius:10px;background:#0b0e14;touch-action:none;display:block;";
        this._ctx = this._canvas.getContext("2d");

        // textareas
        const taWrap = document.createElement("div");
        taWrap.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:6px;";
        this._taA = document.createElement("textarea");
        this._taB = document.createElement("textarea");
        for (const ta of [this._taA, this._taB]){
          ta.spellcheck = false;
          ta.style.cssText = "height:130px;resize:vertical;border-radius:10px;border:1px solid #243041;background:#0b1220;color:#e5e7eb;padding:8px;font:11px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;";
        }
        taWrap.appendChild(this._taA);
        taWrap.appendChild(this._taB);

        this._root.appendChild(bar);
        this._root.appendChild(this._canvas);
        this._root.appendChild(taWrap);
        this.el.appendChild(this._root);

        // state
        this._view = {scale:40, offsetX: 0, offsetY: 0}; // pixels per world unit
        this._mouse = {sx:0, sy:0, wx:0, wy:0};
        this._spaceDown = false;
        this._drag = null; // {which:'A'|'B', poly, idx}
        this._panning = null; // {sx,sy, ox,oy}
        this._drawing = null; // {which, polyIndex}

        this._hoverEdge = null; // {which, key}
        this._hoverT = 0;
        this._hoverTarget = 0;
        this._animating = false;
        this._lastAnim = null;

        // sync from model
        this._syncAllFromModel(true);

        // events
        this._wireEvents();

        // model listeners
        this.model.on("change:polysA", ()=>this._syncPolysFromModel("A"));
        this.model.on("change:polysB", ()=>this._syncPolysFromModel("B"));
        this.model.on("change:active", ()=>{ this._active = this.model.get("active"); this._updateToolbar(); this._syncTextFromState(); this._draw(); });
        this.model.on("change:showA", ()=>{ this._showA = !!this.model.get("showA"); this._chkA.checked=this._showA; this._draw(); });
        this.model.on("change:showB", ()=>{ this._showB = !!this.model.get("showB"); this._chkB.checked=this._showB; this._draw(); });
        this.model.on("change:width", ()=>this._resize());
        this.model.on("change:height", ()=>this._resize());

        this._resize();
        this._fitView();
        this._draw();
      },

      _resize: function(){
        const w = this.model.get("width")|0;
        const h = this.model.get("height")|0;
        const dpr = window.devicePixelRatio || 1;
        this._canvas.style.width = w + "px";
        this._canvas.style.height = h + "px";
        this._canvas.width = Math.floor(w * dpr);
        this._canvas.height = Math.floor(h * dpr);
        this._ctx.setTransform(dpr,0,0,dpr,0,0);
      },

      _syncAllFromModel: function(first){
        this._polysA = toPtObj(this.model.get("polysA"));
        this._polysB = toPtObj(this.model.get("polysB"));
        this._active = this.model.get("active") || "A";
        this._showA = !!this.model.get("showA");
        this._showB = !!this.model.get("showB");
        this._chkA.checked = this._showA;
        this._chkB.checked = this._showB;
        this._updateToolbar();
        this._rebuildEdges();
        this._syncTextFromState(true);
      },

      _syncPolysFromModel: function(which){
        if (which === "A") this._polysA = toPtObj(this.model.get("polysA"));
        else this._polysB = toPtObj(this.model.get("polysB"));
        this._rebuildEdges();
        this._syncTextFromState(true);
        this._draw();
      },

      _rebuildEdges: function(){
        const gA = computeEdgeGroups(this._polysA);
        const gB = computeEdgeGroups(this._polysB);
        this._edgesA = gA.edges; this._groupA = gA.map;
        this._edgesB = gB.edges; this._groupB = gB.map;
      },

      _updateToolbar: function(){
        const active = this._active;
        const styleActive = "border-color:#60a5fa;background:#0a1c35;";
        const styleIdle = "border-color:#334155;background:#0b1220;";
        this._btnA.style.cssText = this._btnA.style.cssText.replace(/border-color:[^;]+;background:[^;]+;/g, "");
        this._btnB.style.cssText = this._btnB.style.cssText.replace(/border-color:[^;]+;background:[^;]+;/g, "");
        this._btnA.style.cssText += (active==="A") ? styleActive : styleIdle;
        this._btnB.style.cssText += (active==="B") ? styleActive : styleIdle;
      },

      _syncTextFromState: function(force){
        // Don't overwrite if the user is actively typing in a box.
        const aFocused = (document.activeElement === this._taA);
        const bFocused = (document.activeElement === this._taB);

        if (force || !aFocused) this._taA.value = toPython(this._polysA);
        if (force || !bFocused) this._taB.value = toPython(this._polysB);
      },

      _wireEvents: function(){
        // buttons
        this._btnA.addEventListener("click", ()=>{
          if (!this._showA){ this.model.set("showA", true); this.model.save_changes(); }
          this.model.set("active","A"); this.model.save_changes();
        });
        this._btnB.addEventListener("click", ()=>{
          if (!this._showB){ this.model.set("showB", true); this.model.save_changes(); }
          this.model.set("active","B"); this.model.save_changes();
        });
        this._chkA.addEventListener("change", ()=>{
          const v = !!this._chkA.checked;
          this.model.set("showA", v); this.model.save_changes();
          if (!v && this._active==="A" && this._showB) { this.model.set("active","B"); this.model.save_changes(); }
        });
        this._chkB.addEventListener("change", ()=>{
          const v = !!this._chkB.checked;
          this.model.set("showB", v); this.model.save_changes();
          if (!v && this._active==="B" && this._showA) { this.model.set("active","A"); this.model.save_changes(); }
        });

        // textarea apply on Enter (Shift+Enter inserts newline)
        this._taA.addEventListener("keydown", (e)=>{
          if (e.key === "Enter" && !e.shiftKey){
            e.preventDefault();
            this._applyText("A");
          }
        });
        this._taB.addEventListener("keydown", (e)=>{
          if (e.key === "Enter" && !e.shiftKey){
            e.preventDefault();
            this._applyText("B");
          }
        });

        // canvas mouse
        this._canvas.addEventListener("contextmenu", (e)=>{ e.preventDefault(); });

        this._canvas.addEventListener("mousemove", (e)=>{
          const rect = this._canvas.getBoundingClientRect();
          const sx = e.clientX - rect.left;
          const sy = e.clientY - rect.top;
          this._mouse.sx=sx; this._mouse.sy=sy;
          const w = this._screenToWorld(sx,sy);
          this._mouse.wx=w.x; this._mouse.wy=w.y;

          if (this._panning){
            const dx = sx - this._panning.sx;
            const dy = sy - this._panning.sy;
            this._view.offsetX = this._panning.ox + dx;
            this._view.offsetY = this._panning.oy + dy;
            this._draw();
            return;
          }

          if (this._drag){
            const polys = (this._drag.which==="A") ? this._polysA : this._polysB;
            const p = polys[this._drag.poly][this._drag.idx];
            p.x = w.x;
            p.y = w.y;
            this._rebuildEdges();
            this._markDirty(this._drag.which);
            this._draw();
            return;
          }

          // hover
          this._updateHover();
          this._draw();
        });

        this._canvas.addEventListener("mousedown", (e)=>{
          e.preventDefault();
          const rect = this._canvas.getBoundingClientRect();
          const sx = e.clientX - rect.left;
          const sy = e.clientY - rect.top;
          const w = this._screenToWorld(sx,sy);
          this._mouse = {sx,sy, wx:w.x, wy:w.y};

          // pan (space+left or middle)
          if ((e.button===0 && this._spaceDown) || e.button===1){
            this._panning = {sx,sy, ox:this._view.offsetX, oy:this._view.offsetY};
            return;
          }

          // right click: draw cycle
          if (e.button === 2){
            this._rightClickDraw();
            return;
          }

          // left click: drag vertex, shift+click deletes vertex
          if (e.button === 0){
            const which = this._active;
            const polys = (which==="A") ? this._polysA : this._polysB;
            if ((which==="A" && !this._showA) || (which==="B" && !this._showB)) return;

            const hit = this._hitPoint(polys, w.x, w.y, 10);
            if (hit){
              if (e.shiftKey){
                // delete vertex
                const ring = polys[hit.poly];
                ring.splice(hit.idx, 1);
                if (ring.length === 0){
                  polys.splice(hit.poly, 1);
                }
                this._drawing = null;
                this._rebuildEdges();
                this._markDirty(which);
                this._draw();
                return;
              }else{
                this._drag = {which: which, poly: hit.poly, idx: hit.idx};
                return;
              }
            }
          }
        });

        this._canvas.addEventListener("mouseup", (e)=>{
          this._drag = null;
          this._panning = null;
          this._draw();
        });
        this._canvas.addEventListener("mouseleave", (e)=>{
          this._drag = null;
          this._panning = null;
          this._hoverTarget = 0;
          this._draw();
        });

        this._canvas.addEventListener("wheel", (e)=>{
          e.preventDefault();
          const w = this.model.get("width")|0;
          const h = this.model.get("height")|0;
          const centerX = w*0.5;
          const centerY = h*0.5;

          const before = this._screenToWorld(centerX, centerY);
          const zoom = Math.exp(-e.deltaY * 0.0015);
          this._view.scale *= zoom;
          this._view.scale = clamp(this._view.scale, 5, 800);
          const after = this._screenToWorld(centerX, centerY);

          // keep center fixed in world
          this._view.offsetX += (before.x - after.x) * this._view.scale;
          this._view.offsetY += -(before.y - after.y) * this._view.scale;

          this._draw();
        }, {passive:false});

        // keyboard
        window.addEventListener("keydown", (e)=>{
          if (e.key === " ") this._spaceDown = true;
          if (e.key === "Escape"){
            if (this._drawing){
              const which = this._drawing.which;
              const polys = (which==="A") ? this._polysA : this._polysB;
              // remove unfinished ring
              polys.splice(this._drawing.polyIndex, 1);
              this._drawing = null;
              this._rebuildEdges();
              this._markDirty(which);
              this._draw();
            }
          }
          if (e.key === "Enter"){
            if (this._drawing){
              this._finishDrawing();
            }
          }
        });
        window.addEventListener("keyup", (e)=>{
          if (e.key === " ") this._spaceDown = false;
        });
      },

      _rightClickDraw: function(){
        const which = this._active;
        if ((which==="A" && !this._showA) || (which==="B" && !this._showB)) return;

        const polys = (which==="A") ? this._polysA : this._polysB;

        if (!this._drawing){
          polys.push([]);
          this._drawing = {which: which, polyIndex: polys.length-1};
        }
        const ring = polys[this._drawing.polyIndex];
        ring.push({x:this._mouse.wx, y:this._mouse.wy});
        this._rebuildEdges();
        this._markDirty(which);
        this._draw();
      },

      _finishDrawing: function(){
        const which = this._drawing.which;
        const polys = (which==="A") ? this._polysA : this._polysB;
        const ring = polys[this._drawing.polyIndex];
        if (!ring || ring.length < 3){
          // discard
          polys.splice(this._drawing.polyIndex, 1);
        }
        this._drawing = null;
        this._rebuildEdges();
        this._markDirty(which);
        this._draw();
      },

      _applyText: function(which){
        const ta = (which==="A") ? this._taA : this._taB;
        const parsed = parsePythonPolys(ta.value);
        if (!parsed.ok){
          this._status.textContent = parsed.err;
          return;
        }
        const newPolys = toPtObj(parsed.polys);
        if (which==="A") this._polysA = newPolys;
        else this._polysB = newPolys;
        this._rebuildEdges();
        this._markDirty(which);
        this._draw();
      },

      _markDirty: function(which){
        this._dirtyWhich = which; // last changed
        this._pendingSave = true;
      },

      _flushModel: function(){
        if (!this._pendingSave) return;
        this._pendingSave = false;

        this.model.set("polysA", toModel(this._polysA));
        this.model.set("polysB", toModel(this._polysB));
        this.model.save_changes();

        // keep text updated
        this._syncTextFromState(false);
      },

      _fitView: function(){
        // fit to union of A+B bounding box, if non-empty
        const pts = [];
        const add = (polys)=>{ for (const r of polys){ for (const p of r){ pts.push(p);} } };
        add(this._polysA); add(this._polysB);
        const w = this.model.get("width")|0;
        const h = this.model.get("height")|0;

        if (pts.length === 0){
          this._view.offsetX = w*0.5;
          this._view.offsetY = h*0.5;
          this._view.scale = 40;
          return;
        }
        let minx=Infinity,maxx=-Infinity,miny=Infinity,maxy=-Infinity;
        for (const p of pts){
          minx=Math.min(minx,p.x); maxx=Math.max(maxx,p.x);
          miny=Math.min(miny,p.y); maxy=Math.max(maxy,p.y);
        }
        const pad = 2;
        const dx = Math.max(1e-6, (maxx-minx)+pad*2);
        const dy = Math.max(1e-6, (maxy-miny)+pad*2);
        const scale = Math.min(w/dx, h/dy) * 0.85;
        this._view.scale = clamp(scale, 5, 800);
        const cx=(minx+maxx)/2, cy=(miny+maxy)/2;
        this._view.offsetX = w/2 - cx*this._view.scale;
        this._view.offsetY = h/2 + cy*this._view.scale;
      },

      _worldToScreen: function(wx, wy){
        const s = this._view.scale;
        return {x: wx*s + this._view.offsetX, y: -wy*s + this._view.offsetY};
      },
      _screenToWorld: function(sx, sy){
        const s = this._view.scale;
        return {x: (sx - this._view.offsetX)/s, y: -(sy - this._view.offsetY)/s};
      },

      _hitPoint: function(polys, wx, wy, radiusPx){
        const rW = radiusPx / this._view.scale;
        const r2 = rW*rW;
        let best = null;
        let bestD = Infinity;
        for (let pi=0; pi<polys.length; pi++){
          const ring = polys[pi];
          for (let i=0; i<ring.length; i++){
            const p = ring[i];
            const d = dist2(wx,wy,p.x,p.y);
            if (d < r2 && d < bestD){
              bestD = d;
              best = {poly:pi, idx:i};
            }
          }
        }
        return best;
      },

      _hitEdge: function(which, wx, wy, radiusPx){
        const rW = radiusPx / this._view.scale;
        const r2 = rW*rW;
        const edges = (which==="A") ? this._edgesA : this._edgesB;
        let best=null, bestD=Infinity;
        for (const e of edges){
          const d = segDist2(wx,wy, e.a.x,e.a.y, e.b.x,e.b.y);
          if (d < r2 && d < bestD){
            bestD = d;
            best = e;
          }
        }
        return best;
      },

      _updateHover: function(){
        let hit = null;
        if (this._showA) hit = this._hitEdge("A", this._mouse.wx, this._mouse.wy, 10);
        let which = "A";
        if (this._showB){
          const hitB = this._hitEdge("B", this._mouse.wx, this._mouse.wy, 10);
          if (!hit && hitB) { hit=hitB; which="B"; }
          else if (hit && hitB){
            // choose closer
            const da = segDist2(this._mouse.wx,this._mouse.wy, hit.a.x,hit.a.y, hit.b.x,hit.b.y);
            const db = segDist2(this._mouse.wx,this._mouse.wy, hitB.a.x,hitB.a.y, hitB.b.x,hitB.b.y);
            if (db < da){ hit=hitB; which="B"; }
          }
        }
        if (hit){
          const newHover = {which:which, key:hit.key};
          if (!this._hoverEdge || this._hoverEdge.which!==newHover.which || this._hoverEdge.key!==newHover.key){
            this._hoverEdge = newHover;
            this._hoverTarget = 1;
            this._startAnim();
          }
        }else{
          if (this._hoverTarget !== 0){
            this._hoverTarget = 0;
            this._startAnim();
          }
        }
      },

      _startAnim: function(){
        if (this._animating) return;
        this._animating = true;
        this._lastAnim = null;
        const step = (ts)=>{
          if (!this._animating) return;
          if (this._lastAnim === null) this._lastAnim = ts;
          const dt = (ts - this._lastAnim) / 1000.0;
          this._lastAnim = ts;
          const speed = 10.0; // per second
          const target = this._hoverTarget;
          if (target > this._hoverT){
            this._hoverT = Math.min(target, this._hoverT + dt*speed);
          }else if (target < this._hoverT){
            this._hoverT = Math.max(target, this._hoverT - dt*speed);
          }
          if (this._hoverT === 0 && this._hoverTarget === 0){
            // clear hover once fully relaxed
            this._hoverEdge = null;
          }
          this._draw();
          if (this._hoverT !== this._hoverTarget){
            requestAnimationFrame(step);
          }else{
            this._animating = false;
          }
        };
        requestAnimationFrame(step);
      },

      _drawArrow: function(p0, p1, ctrl){
        // arrow at mid; ctrl optional {x,y} in world
        let mid;
        if (ctrl){
          // quadratic at t=0.5: 0.25 P0 + 0.5 C + 0.25 P1
          mid = {x: 0.25*p0.x + 0.5*ctrl.x + 0.25*p1.x, y: 0.25*p0.y + 0.5*ctrl.y + 0.25*p1.y};
        }else{
          mid = {x: (p0.x+p1.x)*0.5, y: (p0.y+p1.y)*0.5};
        }
        const s0 = this._worldToScreen(mid.x, mid.y);
        const sA = this._worldToScreen(p0.x, p0.y);
        const sB = this._worldToScreen(p1.x, p1.y);
        const vx = sB.x - sA.x;
        const vy = sB.y - sA.y;
        const L = Math.hypot(vx,vy);
        if (L < 1e-6) return;
        const ux = vx/L, uy = vy/L;
        const size = 7;
        const back = 10;
        const nx = -uy, ny = ux;

        const tipx = s0.x + ux*size;
        const tipy = s0.y + uy*size;
        const bx = s0.x - ux*back;
        const by = s0.y - uy*back;
        const leftx = bx + nx*5;
        const lefty = by + ny*5;
        const rightx = bx - nx*5;
        const righty = by - ny*5;

        const ctx = this._ctx;
        ctx.beginPath();
        ctx.moveTo(tipx,tipy);
        ctx.lineTo(leftx,lefty);
        ctx.lineTo(rightx,righty);
        ctx.closePath();
        ctx.fill();
      },

      _drawEdge: function(which, a, b, key, idxInGroup, groupSize, color, width, alpha){
        const ctx = this._ctx;
        const sA = this._worldToScreen(a.x,a.y);
        const sB = this._worldToScreen(b.x,b.y);

        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.globalAlpha = alpha;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";

        const hover = (this._hoverEdge && this._hoverEdge.which===which && this._hoverEdge.key===key) ? this._hoverT : 0;

        let ctrl = null;
        if (hover > 0 && groupSize > 1){
          // compute control in world using sorted direction (so opposite edges bend same way)
          const k1 = keyPt(a), k2 = keyPt(b);
          let p1=a, p2=b;
          if (k2 < k1){ p1=b; p2=a; }
          const dx = p2.x - p1.x;
          const dy = p2.y - p1.y;
          const L = Math.hypot(dx,dy);
          if (L > 1e-9){
            const nx = -dy/L;
            const ny = dx/L;
            const offsets = offsetList(groupSize, 6); // px
            const offPx = offsets[idxInGroup] * hover;
            const offW = offPx / this._view.scale;
            const mx = (a.x+b.x)*0.5;
            const my = (a.y+b.y)*0.5;
            ctrl = {x: mx + nx*offW, y: my + ny*offW};
          }
        }

        ctx.beginPath();
        ctx.moveTo(sA.x, sA.y);
        if (ctrl){
          const sC = this._worldToScreen(ctrl.x, ctrl.y);
          ctx.quadraticCurveTo(sC.x, sC.y, sB.x, sB.y);
        }else{
          ctx.lineTo(sB.x, sB.y);
        }
        ctx.stroke();

        // arrow
        ctx.fillStyle = color;
        this._drawArrow(a,b,ctrl);

        ctx.restore();
      },

      _draw: function(){
        // model flush (at most once per frame)
        this._flushModel();

        const ctx = this._ctx;
        const w = this.model.get("width")|0;
        const h = this.model.get("height")|0;
        ctx.clearRect(0,0,w,h);

        // background
        ctx.save();
        ctx.fillStyle = "#0b0e14";
        ctx.fillRect(0,0,w,h);
        ctx.restore();

        // draw datasets
        const active = this._active;
        const showA = this._showA;
        const showB = this._showB;

        const drawSet = (which)=>{
          const polys = (which==="A") ? this._polysA : this._polysB;
          const edges = (which==="A") ? this._edgesA : this._edgesB;
          const groups = (which==="A") ? this._groupA : this._groupB;

          const isActive = (which === active);
          const baseColor = (which==="A") ? "#60a5fa" : "#f59e0b";
          const color = baseColor;
          const alpha = isActive ? 0.95 : 0.35;
          const width = isActive ? 2.6 : 2.0;

          // edges
          for (const e of edges){
            const grp = groups[e.key] || [];
            const idx = grp.indexOf(e);
            const gi = (idx>=0)? idx : 0;
            this._drawEdge(which, e.a, e.b, e.key, gi, grp.length, color, width, alpha);
          }

          // vertices
          ctx.save();
          ctx.globalAlpha = isActive ? 0.95 : 0.35;
          ctx.fillStyle = color;
          for (const ring of polys){
            for (const p of ring){
              const s = this._worldToScreen(p.x,p.y);
              ctx.beginPath();
              ctx.arc(s.x,s.y,3.5,0,Math.PI*2);
              ctx.fill();
            }
          }
          ctx.restore();
        };

        if (showA) drawSet("A");
        if (showB) drawSet("B");

        // drawing preview
        if (this._drawing){
          const which = this._drawing.which;
          const polys = (which==="A") ? this._polysA : this._polysB;
          const ring = polys[this._drawing.polyIndex];
          if (ring && ring.length >= 1){
            const last = ring[ring.length-1];
            const s1 = this._worldToScreen(last.x,last.y);
            const s2 = this._worldToScreen(this._mouse.wx,this._mouse.wy);
            ctx.save();
            ctx.strokeStyle = "#cbd5e1";
            ctx.globalAlpha = 0.8;
            ctx.setLineDash([6,6]);
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(s1.x,s1.y);
            ctx.lineTo(s2.x,s2.y);
            ctx.stroke();
            ctx.restore();
          }
        }

        // status
        const z = (this._view.scale/40).toFixed(2) + "Ã—";
        this._status.textContent = "active " + active + " â€¢ zoom " + z;
      }
    });

    return {PolyDiffView: PolyDiffView};
  });
})();
"""


def _ensure_js_loaded() -> None:
    display(Javascript(_JS))


def poly_diff(
    A: Optional[Union[PolyVar, Polys]] = None,
    B: Optional[Union[PolyVar, Polys]] = None,
    *,
    width: int = 560,
    height: int = 320,
    showA: bool = True,
    showB: bool = True,
) -> PolyDiffWidget:
    """
    Create the diff editor widget for two polygon sets A and B.

    Pass either:
      - PolyVar instances (recommended for real two-way sync), or
      - plain list-of-rings (UI edits update the widget state; reading is via widget.polysA/B)

    Returns the widget (display it in a notebook cell).
    """
    _ensure_js_loaded()

    w = PolyDiffWidget(width=width, height=height, showA=showA, showB=showB)

    links = []

    def _coerce(polys: Optional[Union[PolyVar, Polys]]) -> Optional[Polys]:
        if polys is None:
            return None
        if isinstance(polys, PolyVar):
            return polys.value  # type: ignore[return-value]
        return polys  # type: ignore[return-value]

    # init
    if A is not None and not isinstance(A, PolyVar):
        w.polysA = A
    if B is not None and not isinstance(B, PolyVar):
        w.polysB = B

    # link if PolyVar
    if isinstance(A, PolyVar):
        links.append(traitlets.link((A, "value"), (w, "polysA")))
    if isinstance(B, PolyVar):
        links.append(traitlets.link((B, "value"), (w, "polysB")))

    # keep links alive
    w._links = links  # type: ignore[attr-defined]
    return w