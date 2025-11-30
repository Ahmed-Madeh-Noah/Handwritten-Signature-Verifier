document.addEventListener('DOMContentLoaded', () => {

    const toggleBtn = document.getElementById('theme-toggle');
    const currentTheme = localStorage.getItem('theme');

    if (currentTheme) {
        document.documentElement.setAttribute('data-theme', currentTheme);
        updateIcon(currentTheme);
    }

    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            let theme = document.documentElement.getAttribute('data-theme');
            let newTheme = theme === 'dark' ? 'light' : 'dark';

            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            updateIcon(newTheme);
        });
    }

    function updateIcon(theme) {
        if (toggleBtn) {
            toggleBtn.textContent = theme === 'dark' ? '☀️' : '🌙';
        }
    }

    setupImagePreview('sig1', 'preview-1');
    setupImagePreview('sig2', 'preview-2');
});

function setupImagePreview(inputId, imgTargetId) {
    const input = document.getElementById(inputId);
    const imgTarget = document.getElementById(imgTargetId);

    if (input && imgTarget) {
        input.addEventListener('change', function (e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();

                reader.onload = function (e) {
                    imgTarget.src = e.target.result;
                    imgTarget.style.display = 'block';
                    imgTarget.style.border = '2px solid var(--color-teal)';
                }

                reader.readAsDataURL(file);
            }
        });
    }
}