function playVids(e) 
{ 
    var t = document.getElementById(e + "Merge"), i = document.getElementById(e), n = .5, o = i.videoWidth / 2, l = i.videoHeight, a = t.getContext("2d"); 
    if (i.readyState > 3) 
    { 
        function d(e) 
        { bcr = t.getBoundingClientRect(), 
            n = (e.touches[0].pageX - bcr.x) / bcr.width 
        } 
        i.play(), 
        t.addEventListener("mousemove", (
            function (e) 
            { 
                bcr = t.getBoundingClientRect(), 
                n = (e.pageX - bcr.x) / bcr.width 
            }), !1), 
            t.addEventListener("touchstart", d, !1), 
            t.addEventListener("touchmove", d, !1), 
            requestAnimationFrame((
                function e() 
                { 
                    a.drawImage(i, 0, 0, o, l, 0, 0, o, l); 
                    var t = (o * n).clamp(0, o), d = (o - o * n).clamp(0, o); 
                    a.drawImage(i, t + o, 0, d, l, t, 0, d, l), requestAnimationFrame(e); 
                    var r = .09 * l, c = .025 * l, h = .04 * l, m = l / 10, g = .007 * l, u = o * n; 
                    a.arc(u, m, .7 * r, 0, 2 * Math.PI, !1), a.fillStyle = "#FFD79340", 
                    a.fill(), a.beginPath(), a.moveTo(o * n, 0), a.lineTo(o * n, l), a.closePath(), 
                    a.strokeStyle = "#444444", a.lineWidth = .3, a.stroke(), a.beginPath(), 
                    a.moveTo(u, m - g / 2), a.lineTo(u + r / 2 - h / 2, m - g / 2), 
                    a.lineTo(u + r / 2 - h / 2, m - c / 2), a.lineTo(u + r / 2, m), 
                    a.lineTo(u + r / 2 - h / 2, m + c / 2), a.lineTo(u + r / 2 - h / 2, m + g / 2), 
                    a.lineTo(u - r / 2 + h / 2, m + g / 2), a.lineTo(u - r / 2 + h / 2, m + c / 2), 
                    a.lineTo(u - r / 2, m), a.lineTo(u - r / 2 + h / 2, m - c / 2), a.lineTo(u - r / 2 + h / 2, m), 
                    a.lineTo(u - r / 2 + h / 2, m - g / 2), a.lineTo(u, m - g / 2), a.closePath(), 
                    a.fillStyle = "#444444", a.fill() 
                })) 
            } 
        } 
function resizeAndPlay(e) 
{ 
    var t = document.getElementById(e.id + "Merge"); 
    t.width = e.videoWidth / 2, t.height = e.videoHeight, e.play(), e.style.height = "0px", playVids(e.id) 
} 
Number.prototype.clamp = function (e, t) 
    { 
        return Math.min(Math.max(this, e), t) 
    };