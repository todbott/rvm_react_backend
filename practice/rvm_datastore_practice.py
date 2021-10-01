# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 16:54:56 2021

@author: Gillies
"""

from google.oauth2 import service_account
from google.cloud import ndb

key_location = "hotaru-kanri-38df8cf3173f.json"
credentials = service_account.Credentials.from_service_account_file(key_location)

ndbclient = ndb.Client(project="hotaru-kanri", credentials=credentials)


class rvmRecord(ndb.Model):
    humanText = ndb.StringProperty()
    aiText = ndb.StringProperty()
    imageName = ndb.StringProperty()
     
def createRvmRecord(h, a, i):
    one = rvmRecord(
        humanText = h,
        aiText = a,
        imageName = i)
    one.put()
    
with ndbclient.context():
    createRvmRecord("This is todd", "This is a computer", "920")