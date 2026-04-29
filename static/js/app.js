// 현재 페이지 네비게이션 하이라이트
document.addEventListener('DOMContentLoaded', () => {
  const path = window.location.pathname;
  document.querySelectorAll('.nav-links a').forEach(a => {
    if (a.getAttribute('href') === path) {
      a.style.color = 'var(--primary)';
      a.style.fontWeight = '600';
    }
  });
});
