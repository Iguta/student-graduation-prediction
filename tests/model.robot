*** Settings ***
Library    OperatingSystem

*** Variables ***
${MODEL_PATH}    ${CURDIR}/../model.pkl
${SCALER_PATH}    ${CURDIR}/../scaler.pkl

*** Test Cases ***
Pickle File for the Model Exist
    File Should Exist    ${MODEL_PATH}    Model file does not exist
    File Should Exist    ${SCALER_PATH}    Scaler file does not exist

Test that Picke File is NOT empty
    ${model_size}=    Get File Size    ${MODEL_PATH}
    Should Be True    ${model_size} > 0    Model file is empty
    ${scaler_size}=    Get File Size    ${SCALER_PATH}
    Should Be True    ${scaler_size} > 0    Scaler file is empty