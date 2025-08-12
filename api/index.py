from mangum import Mangum
from app.index import app

handler = Mangum(app)
