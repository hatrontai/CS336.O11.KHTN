These modules are required to create vector space model
Define n = number of dimension in vector space
```python
def get_embed_dim():
	...
	return n
```
```python
def image_encoder(image -> PIL Image):
	...
	return embedding -> numpy array
```
```python
def text_encoder(text -> string):
	...
	return embedding -> numpy array
```
