*** Settings ***
Library    SeleniumLibrary

*** Variables ***
${URL}      https://student-graduation-prediction.streamlit.app/
${BROWSER}  Chrome

*** Test Cases *

Home Page Loads
    [Tags]    UI Tests
    Open Browser    ${URL}    ${BROWSER}
    Wait Until Page Contains   Student Dropout
    [Teardown]    Close Browser
Data Oview Page Load
    [Tags]    UI Tests 
    Open Browser    ${URL}    ${BROWSER}
    Wait Until Page Contains Element    xpath=//*[@id="dataset-overview"]    timeout=10s

Data Analysis Page Loads
    [Tags]    current
    Open Browser    ${URL}    ${BROWSER}
    Wait Until Page Contains Element    xpath=//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[2]/div/div/div/div/div/div/div/label[2]/div[2]/div/p
    Click Element    xpath=//*[@id="root"]/div[1]/div[1]/div/div/div/section[1]/div[1]/div[2]/div/div/div/div/div/div/div/label[2]/div[2]/div/p
