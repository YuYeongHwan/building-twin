from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["pages"])
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/buildings", response_class=HTMLResponse)
def buildings_page(request: Request):
    return templates.TemplateResponse("buildings.html", {"request": request})


@router.get("/inspect", response_class=HTMLResponse)
def inspect_page(request: Request):
    return templates.TemplateResponse("inspect.html", {"request": request})


@router.get("/result/{inspection_id}", response_class=HTMLResponse)
def result_page(request: Request, inspection_id: int):
    return templates.TemplateResponse("result.html", {"request": request})
