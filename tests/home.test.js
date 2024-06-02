const { test, expect } = require('@playwright/test');

test('Check form fields and submission', async ({ page }) => {
    // Navigate to the page
    await page.goto('http://localhost:5000/');

    // Check if the form fields are present
    const ageInput = await page.$('input[name="age"]');
    expect(ageInput).not.toBeNull();

    const genderSelect = await page.$('select[name="gender"]');
    expect(genderSelect).not.toBeNull();

    const titleSelect = await page.$('select[name="name_title"]');
    expect(titleSelect).not.toBeNull();

    const sibspSelect = await page.$('select[name="sibsp"]');
    expect(sibspSelect).not.toBeNull();

    const pclassSelect = await page.$('select[name="pclass"]');
    expect(pclassSelect).not.toBeNull();

    const embarkedSelect = await page.$('select[name="embarked"]');
    expect(embarkedSelect).not.toBeNull();

    const cabinSelect = await page.$('select[name="cabin_multiple"]');
    expect(cabinSelect).not.toBeNull();

    // Fill out the form
    await page.fill('input[name="age"]', '42');
    await page.selectOption('select[name="gender"]', { label: 'Female' });
    await page.selectOption('select[name="name_title"]', { label: 'Mrs' });
    await page.selectOption('select[name="sibsp"]', { label: 'None' });
    await page.selectOption('select[name="pclass"]', { label: 'First' });
    await page.selectOption('select[name="embarked"]', { label: 'Cherbourg' });
    await page.selectOption('select[name="cabin_multiple"]', { label: '2 person cabin' });

    // Submit the form
    await page.click('input[type="submit"]');

    // Check if the result is displayed
    const resultText = await page.textContent('h2');
    expect(resultText).toContain('Survived with a probability of 83.6%');
});
