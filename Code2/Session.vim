let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/Dissertation/Code2
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +117 Agent/QuantileAgent.py
badd +1 MultiAgent/MeanField.py
badd +1 main.py
badd +57 Environment/Bravais.py
badd +0 NetrwTreeListing\ 1
badd +0 Agent/ResidueAgent.py
argglobal
silent! argdel *
edit MultiAgent/MeanField.py
set splitbelow splitright
wincmd _ | wincmd |
split
1wincmd k
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd w
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
2wincmd h
wincmd w
wincmd w
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe '1resize ' . ((&lines * 31 + 33) / 66)
exe 'vert 1resize ' . ((&columns * 135 + 135) / 271)
exe '2resize ' . ((&lines * 31 + 33) / 66)
exe 'vert 2resize ' . ((&columns * 135 + 135) / 271)
exe '3resize ' . ((&lines * 32 + 33) / 66)
exe 'vert 3resize ' . ((&columns * 15 + 135) / 271)
exe '4resize ' . ((&lines * 32 + 33) / 66)
exe 'vert 4resize ' . ((&columns * 165 + 135) / 271)
exe '5resize ' . ((&lines * 32 + 33) / 66)
exe 'vert 5resize ' . ((&columns * 89 + 135) / 271)
argglobal
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 25 - ((24 * winheight(0) + 15) / 31)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
25
normal! 045|
wincmd w
argglobal
if bufexists('Agent/ResidueAgent.py') | buffer Agent/ResidueAgent.py | else | edit Agent/ResidueAgent.py | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 121 - ((30 * winheight(0) + 15) / 31)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
121
normal! 0
wincmd w
argglobal
if bufexists('NetrwTreeListing') | buffer NetrwTreeListing | else | edit NetrwTreeListing | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 1 - ((0 * winheight(0) + 16) / 32)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
1
normal! 0
lcd ~/Dissertation/Code2
wincmd w
argglobal
if bufexists('~/Dissertation/Code2/Environment/Bravais.py') | buffer ~/Dissertation/Code2/Environment/Bravais.py | else | edit ~/Dissertation/Code2/Environment/Bravais.py | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 40 - ((0 * winheight(0) + 16) / 32)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
40
normal! 04|
lcd ~/Dissertation/Code2
wincmd w
argglobal
if bufexists('~/Dissertation/Code2/main.py') | buffer ~/Dissertation/Code2/main.py | else | edit ~/Dissertation/Code2/main.py | endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let s:l = 82 - ((13 * winheight(0) + 16) / 32)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
82
normal! 0
lcd ~/Dissertation/Code2
wincmd w
2wincmd w
exe '1resize ' . ((&lines * 31 + 33) / 66)
exe 'vert 1resize ' . ((&columns * 135 + 135) / 271)
exe '2resize ' . ((&lines * 31 + 33) / 66)
exe 'vert 2resize ' . ((&columns * 135 + 135) / 271)
exe '3resize ' . ((&lines * 32 + 33) / 66)
exe 'vert 3resize ' . ((&columns * 15 + 135) / 271)
exe '4resize ' . ((&lines * 32 + 33) / 66)
exe 'vert 4resize ' . ((&columns * 165 + 135) / 271)
exe '5resize ' . ((&lines * 32 + 33) / 66)
exe 'vert 5resize ' . ((&columns * 89 + 135) / 271)
tabnext 1
if exists('s:wipebuf') && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 winminheight=1 winminwidth=1 shortmess=filnxtToOFc
let s:sx = expand("<sfile>:p:r")."x.vim"
if file_readable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &so = s:so_save | let &siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
