## Some jpg-images cannot be read

When I try to read an image from Wiki dataset I get the following error:

```python
from PIL import Image
im = Image.open('../../_input/style//915.jpg')
```

local variable 'photoshop' referenced before assignment

[**Solution**](https://github.com/python-pillow/Pillow/pull/3771)