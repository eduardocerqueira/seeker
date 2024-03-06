#date: 2024-03-06T17:07:11Z
#url: https://api.github.com/gists/9b4f211089b0275d2bc6766d6ea66f61
#owner: https://api.github.com/users/fauzaanu

@hooks.register('insert_global_admin_js', order=2)
def global_admin_js():
    return """
<script>
   // add a bottom button to the admin to switch to RTL and back
    const rtl_button = document.createElement('button');
    rtl_button.innerHTML = 'RTL';
    rtl_button.onclick = () => {
        const html = document.querySelector('html');
        html.dir = html.dir === 'rtl' ? 'ltr' : 'rtl';
    }
    
    // add the button to the bottom of main
    const main_doc = document.querySelector('#main');
    document.querySelector('body').insertBefore(rtl_button, main_doc);
    
    // center the button horizontally
    rtl_button.style.position = 'fixed';
    rtl_button.style.bottom = '0';
    rtl_button.style.left = '50%';
    rtl_button.style.transform = 'translateX(-50%)';
    rtl_button.style.zIndex = '1000';
</script>
"""