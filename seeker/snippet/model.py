#date: 2023-07-03T17:00:08Z
#url: https://api.github.com/gists/271f855cf58d280bccf3cf433da85395
#owner: https://api.github.com/users/autominds

import datetime
from sqlalchemy import Boolean, Column, String, Date, func, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base
from sqlalchemy import MetaData

from uuid import uuid4

metadata = MetaData()

_Base = declarative_base(metadata=metadata)
metadata = _Base.metadata

class Base(_Base):
    __abstract__ = True
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)  # default=uuid4,
    created_at = Column(DateTime, comment="创建时间", server_default=func.now())  # 创建时间default=func.now(),
    modified_at = Column(DateTime, onupdate=func.now(), comment="修改时间", server_default=func.now())  # 修改时间
    is_active = Column(Boolean, default=True, comment="是否启用")
    remark = Column(String, default="", comment="备注")  # 备注