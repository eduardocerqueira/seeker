#date: 2025-10-07T17:07:37Z
#url: https://api.github.com/gists/5d6a1c4fe514775e837b2f6bc628a729
#owner: https://api.github.com/users/AmrishJhingoer

from IPython.display import display,HTML
c1,c2,f1,f2,fs1,fs2=\
'#eb3434','#eb3446','Akronim','Smokum',30,15
def dhtml(string,fontcolor=c1,font=f1,fontsize=fs1):
    display(HTML("""<style>
    @import 'https://fonts.googleapis.com/css?family="""\
    +font+"""&effect=3d-float';</style>
    <h1 class='font-effect-3d-float' style='font-family:"""+\
    font+"""; color:"""+fontcolor+"""; font-size:"""+\
    str(fontsize)+"""px;'>%s</h1>"""%string))


dhtml('Here is some cool text.<br>' \
'Styled for the notebook. Try to run me.' )