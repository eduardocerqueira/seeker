//date: 2023-12-26T17:00:12Z
//url: https://api.github.com/gists/0f686c0fadbd3c52121dc910b849ed3e
//owner: https://api.github.com/users/egtann

// setCSP secures the user against XSS attacks. Since we use inline styles and  
// scripts, this applies a cryptographically random 16-byte nonce to the        
// context for the browser to verify inline scripts and source tags.            
func setCSP(next http.Handler) http.Handler {                                   
        fn := func(w http.ResponseWriter, r *http.Request) {                    
                byt := make([]byte, 16)                                         
                _, err := rand.Read(byt)                                        
                if err != nil {                                                 
                        err = fmt.Errorf("read: %w", err)                       
                        http.Error(w, err.Error(),                              
                                http.StatusInternalServerError)                    
                        return                                                  
                }                                                               
                nonce := base64.URLEncoding.EncodeToString(byt)                 
                ctx := context.WithValue(r.Context(), app.NonceKey, nonce)       
                csp := []string{                                                
                        "default-src 'self'",                                   
                        fmt.Sprintf("script-src 'self' 'nonce-%s'", nonce),             
                        fmt.Sprintf("style-src 'self' 'nonce-%s'", nonce),                                      
                }                                                               
                h := w.Header()                                                 
                h.Set("Content-Security-Policy", strings.Join(csp, "; "))          
                next.ServeHTTP(w, r.WithContext(ctx))                           
        }                                                                       
        return http.HandlerFunc(fn)                                             
}

// setNonce overrides the nonce in the context to match the one provided by the 
// client. This enables us to re-use the same nonce on subsequent htmx ajax        
// requests as long as we're on the same page.                                                                    
func setNonce(next http.Handler) http.Handler {                                 
        fn := func(w http.ResponseWriter, r *http.Request) {                    
                if nonce := r.Header.Get("X-Nonce"); nonce != "" {              
                        ctx := context.WithValue(r.Context(), app.NonceKey, nonce)
                        next.ServeHTTP(w, r.WithContext(ctx))                   
                        return                                                  
                }                                                               
                next.ServeHTTP(w, r)                                            
        }                                                                       
        return http.HandlerFunc(fn)                                             
}