function removePlaceholderOption(selectElement) {
    // Get the placeholder option
    var placeholderOption = selectElement.querySelector('.placeholder-option');

    // Remove the placeholder option from the dropdown list
    if (placeholderOption) {
        placeholderOption.remove();
    }
}

// Smooth scroll to anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();

        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Toggle modal visibility
const modalBtn = document.querySelector('.modal-btn');
const modal = document.querySelector('.modal');

modalBtn.addEventListener('click', () => {
    modal.classList.toggle('show-modal');
});

// Close modal when clicking outside
window.addEventListener('click', (e) => {
    if (e.target === modal) {
        modal.classList.remove('show-modal');
    }
});

// Close modal when pressing Escape key
window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.classList.contains('show-modal')) {
        modal.classList.remove('show-modal');
    }
});