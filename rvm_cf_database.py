# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 17:15:30 2021

@author: Gillies
"""

from google.cloud import ndb

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
    

def cors_enabled_function(request):
    
    ndbclient = ndb.Client()

    request_json = request.get_json(silent=True)
    request_args = request.args

    # For more information about CORS and CORS preflight requests, see:
    # https://developer.mozilla.org/en-US/docs/Glossary/Preflight_request

    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }

        return ('', 204, headers)

    else:

        if request_json and 'humanText' in request_json:
            h = request_json['humanText']
            a = request_json['aiText']
            i = request_json['imageName']
        elif request_args and 'humanText' in request_args:
            h = request_args('humanText')
            a = request_args('aiText')
            i = request_args('imageName')



        with ndbclient.context():
            createRvmRecord(h, a, i)

        
        
    

        # Set CORS headers for the main request
        headers = {
            'Access-Control-Allow-Origin': '*'
        }
            
        
        return ("Uploaded", 200, headers)

