#date: 2022-02-16T17:07:30Z
#url: https://api.github.com/gists/a5fe1cebd5aab4c5de31cf9ef5f0f29a
#owner: https://api.github.com/users/gileshall

import os
import fnmatch
from zipfile import is_zipfile, ZipFile
from io import StringIO, BytesIO
import json
import firecloud.api
from pprint import pprint
from functools import lru_cache

def glob_find(pattern=None, items=None, is_unique=True):
    if pattern[0] != '*':
        pattern = '*' + pattern
    if pattern[-1] != '*':
        pattern = pattern + '*'

    out = fnmatch.filter(items, pattern)
    if is_unique:
        if len(out) > 1:
            msg = f"{pattern} matches {len(out)} items, when it can only match one"
            raise ValueError(msg)
        return out[0]
    return out

  
class WorkspaceTable(object):
    def __init__(self, workspace, table_name, attributes=None, table_id=None):
        self.workspace = workspace
        self.table_name = table_name
        self.attributes = tuple(attributes) if attributes else tuple()
        self.table_id = table_id or f"{self.table_name}_id"
    
    def __repr__(self):
        attrs = list(self.attributes)
        return \
            f"{self.__class__.__name__}(" + \
            f"workspace='{self.workspace.workspace_name}', " + \
            f"table_name='{self.table_name}', " + \
            f"attributes={repr(attrs)}, " + \
            f"table_id='{self.table_id}'" + \
            f")"
       
class DiskCache(object):
    _cache = None

    def __init__(self, cache=None):
        cls = self.__class__
        self.cache_fn = cache_fn
        if cache is not None:
            cls.cache = cache
        elif cls.cache is None:
            cls.cache = {}

    @classmethod
    def load(cls, cache_fn):
        with open(cache_fn, 'rb') as fh:
            cache = json.load(fh)
        return cls(cache=cache)

    def save(self, cache_fn):
        with open(cache_fn, 'wb') as fh:
            json.dump(self.cache, fh)

    def __contains__(self, key):
        key = json.dumps(key)
        return key in self._cache

    def get(self, key):
        key = json.dumps(key)
        return self._cache.get(key, self.sentinel)

class CallCache(object):
    _cache = None

    def __init__(self, callname, fqid):
        self.callname = callname
        self.fqid = fqid
        if self._cache is None:
            self._cache = {}

    def _cache_get(self, key):
        return self._cache.get(key)
    
    @classmethod
    @lru_cache(maxsize=None)
    def api_call(cls, call=None, to_json=True):
        print("cache miss", to_json)
        (funcname, args, kw) = json.loads(call)
        attr = getattr(firecloud.api, funcname)
        ret = attr(*args, **kw)
        if to_json:
            ret = ret.json()
        return ret

    @classmethod
    def cache_clear(cls):
        cls.api_call.cache_clear()

    def __call__(self, *args, to_json=True, **kw):
        args = list(self.fqid) + list(args)
        call = (self.callname, args, kw)
        call = json.dumps(call)
        ret = self.api_call(call=call, to_json=to_json)
        return ret

class ApiWrapper(object):
    def __init__(self, fqid=None):
        self.fqid = fqid

    def __getattr__(self, name):
        attr = getattr(firecloud.api, name)
        if not callable(attr):
            return attr
        return CallCache(name, self.fqid)
        
class FirecloudObject(object):
    @property
    def api(self):
        return ApiWrapper(self._get_fqid())

    def _get_fqid(self):
        return tuple()

class Workspace(FirecloudObject):
    def __init__(self, workspace_name=None, billing_project=None):
        self.workspace_name = workspace_name
        self.billing_project = billing_project
        self._load_tables()

    @classmethod
    def get_workspace(cls, lookup=None, key="name"):
        ws_list = firecloud.api.list_workspaces().json()
        ws_map = {it["workspace"][key]: it for it in ws_list}
        ws_names = list(ws_map.keys())
        ws_match = glob_find(lookup, ws_names)
        ws = ws_map[ws_match]
        ws_name = ws["workspace"]["name"]
        ws_namespace = ws["workspace"]["namespace"]
        return Workspace(ws_name, ws_namespace)
    
    def _get_fqid(self):
        return (self.billing_project,self.workspace_name)
    
    def get_submission(self, lookup=None, key="submissionId"):
        sub_list = self.api.list_submissions()
        sub_map = {it[key]: it for it in sub_list}
        sub_names = list(sub_map.keys())
        sub_match = glob_find(lookup, sub_names)
        sub = sub_map[sub_match]
        sub_id = sub["submissionId"]
        return Submission(workspace=self, submission_id=sub_id)
        
    def _load_tables(self):
        self.tables = {}
        tables = self.api.list_entity_types()
        
        for table_name in tables:
            table = tables[table_name]
            attrs = table["attributeNames"]
            table_id = table["idName"]
            ws_table = WorkspaceTable(
                workspace=self,
                table_name=table_name,
                attributes=attrs,
                table_id=table_id
            )
            self.tables[table_name] = ws_table
                    
    def get_entities_as_dataframe(self, table_name):
        resp = self.get_entities_tsv(table_name, model="flexible")
        fh = BytesIO(resp.content)
        tsv_list = []
        if is_zipfile(fh):
            with ZipFile(fh) as zf:
                for zinfo in zf.infolist():
                    assert not zinfo.is_dir()
                    tsv = zf.read(zinfo).decode()
                    tsv_list.append(tsv)
        else:
            tsv_list.append(resp.text)
        frames = [pd.read_csv(StringIO(tsv), sep='\t') for tsv in tsv_list]
        return frames

class Submission(FirecloudObject):
    def __init__(self, workspace=None, submission_id=None):
        self.workspace = workspace
        self.submission_id = submission_id
    
    def _get_fqid(self):
        fqid = self.workspace._get_fqid()
        return fqid + (self.submission_id, )
    
    @property
    def workflows(self):
        sub_info = self.api.get_submission()
        cstr = lambda wid: Workflow(submission=self, workflow_id=wid)
        wf_list = [cstr(wf["workflowId"]) for wf in sub_info["workflows"]]
        return wf_list

class Workflow(FirecloudObject):
    def __init__(self, submission=None, workflow_id=None, parent_workflow=None):
        self.submission = submission
        self.workflow_id = workflow_id
        self.parent_workflow = parent_workflow

    def _get_fqid(self):
        fqid = self.submission._get_fqid()
        return fqid + (self.workflow_id, )
    
    @property
    def metadata(self):
        return self.api.get_workflow_metadata()
    
    @property
    def calls(self):
        calls = self.metadata["calls"]
        for callname in calls:
            print(callname)
            steps = []
            for step in calls[callname]:
                if "subWorkflowId" in step:
                    sub_wid = step["subWorkflowId"]
                    print(sub_wid)
                    sub = self.__class__(self.submission, sub_wid)
                    step["subWorkflow"] = sub.calls
                steps.append(step)
            calls[callname] = steps
        return calls
    
ws = Workspace.get_workspace("- gh -")
sub = ws.get_submission("43b")
wf = sub.workflows[0]
c = wf.calls
pprint(c)
