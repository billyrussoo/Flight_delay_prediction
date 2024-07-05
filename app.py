import streamlit as st
import pandas as pd
import pickle

# models
with open('xgboost_model.pkl', 'rb') as f1:
    classification_model = pickle.load(f1)

with open('regression_model.pkl', 'rb') as f2:
    regression_model = pickle.load(f2)


def load_unique_values():
    unique_airline_codes = ['UA', 'DL', 'NK', 'WN', 'AA', 'YX', 'AS', 'B6', 'OH', 'G4', 'EV', 'OO', '9E', 'MQ', 'F9', 'YV', 'QX', 'HA']
    unique_origins = ['FLL', 'MSP', 'DEN', 'MCO', 'DAL', 'DCA', 'HSV', 'IAH', 'SEA', 'ATL', 'RDU', 'MDW', 'BDL', 'BWI', 'STT', 'SRQ', 'JFK', 'GRR', 'DFW', 'CLT', 'ORD', 'LAS', 'TUL', 'USA', 'SLC', 'BNA', 'AUS', 'IND', 'MHT', 'SFO', 'PRC', 'BOS', 'LAX', 'SMF', 'DTW', 'SAT', 'MSY', 'CMH', 'STL', 'SJU', 'PHX', 'TPA', 'LGA', 'PHL', 'GRK', 'ILM', 'JLN', 'MKE', 'BIL', 'BLI', 'CHS', 'RIC', 'GSO', 'MCI', 'EWR', 'ELP', 'SDF', 'HPN', 'SAN', 'BHM', 'SJC', 'ASE', 'HNL', 'MAF', 'BUF', 'TUS', 'SYR', 'MSN', 'FAT', 'FWA', 'LAN', 'ABI', 'ICT', 'CMI', 'OAK', 'IAD', 'EUG', 'MIA', 'CHA', 'BTV', 'RSW', 'PDX', 'SNA', 'PIT', 'OGG', 'HRL', 'BIS', 'SHR', 'BOI', 'RNO', 'GSP', 'MLB', 'GRB', 'CRW', 'LGB', 'OKC', 'PVD', 'YAK', 'CID', 'FSD', 'HYS', 'VPS', 'ROC', 'SAV', 'CLE', 'RDD', 'ANC', 'DVL', 'SWO', 'BUR', 'PIR', 'HOU', 'PBI', 'MLI', 'AZA', 'MKG', 'PNS', 'MYR', 'GTF', 'LSE', 'RDM', 'ORF', 'KOA', 'MQT', 'COS', 'CVG', 'MOB', 'AVL', 'BTR', 'JNU', 'CDC', 'SPS', 'ECP', 'JAX', 'SBN', 'RAP', 'AMA', 'JMS', 'SFB', 'ABQ', 'ALB', 'BGM', 'MGM', 'MEM', 'ONT', 'TVC', 'GEG', 'OMA', 'ISP', 'DLH', 'PIE', 'PSP', 'ACY', 'BJI', 'LBB', 'SHV', 'TTN', 'PIA', 'GNV', 'PWM', 'BZN', 'BRW', 'MFR', 'GPT', 'ATW', 'DSM', 'CKB', 'SGU', 'SBA', 'OTZ', 'BGR', 'XNA', 'CAE', 'SGF', 'PGD', 'VLD', 'CPR', 'LIH', 'JAN', 'OAJ', 'ABE', 'EGE', 'CRP', 'FNT', 'WRG', 'FCA', 'CHO', 'MRY', 'SCE', 'AEX', 'MCW', 'RST', 'SIT', 'BMI', 'ACV', 'GCC', 'LFT', 'ITO', 'ABY', 'GRI', 'BPT', 'APN', 'TLH', 'FSM', 'ALO', 'FAY', 'GJT', 'IAG', 'FLG', 'LBE', 'CAK', 'BFL', 'VEL', 'STX', 'TYS', 'EYW', 'AVP', 'KTN', 'EVV', 'BRO', 'SWF', 'LIT', 'GGG', 'MSO', 'PSC', 'SLN', 'SBP', 'BIH', 'MLU', 'HOB', 'DAY', 'DBQ', 'ROW', 'LEX', 'PIH', 'EWN', 'ESC', 'SUX', 'PAE', 'CNY', 'CDV', 'COU', 'LBF', 'MFE', 'PSE', 'LAR', 'FAI', 'SCC', 'MTJ', 'SJT', 'AGS', 'MDT', 'ELM', 'HDN', 'MOT', 'IDA', 'FAR', 'GFK', 'JAC', 'LAW', 'SUN', 'IMT', 'LCH', 'CLL', 'PHF', 'TRI', 'BET', 'PBG', 'BQN', 'BRD', 'DAB', 'DRO', 'STS', 'ROA', 'DDC', 'INL', 'ORH', 'TXK', 'CWA', 'BFF', 'DHN', 'SCK', 'LRD', 'BTM', 'IPT', 'PIB', 'HYA', 'PVU', 'LBL', 'AZO', 'ACK', 'YUM', 'DRT', 'COD', 'LNK', 'SHD', 'CIU', 'HLN', 'LWB', 'MEI', 'SAF', 'CYS', 'ABR', 'LCK', 'RKS', 'GCK', 'LWS', 'EAR', 'GTR', 'CSG', 'RFD', 'BLV', 'TYR', 'CMX', 'PAH', 'JST', 'TOL', 'PSG', 'MHK', 'YKM', 'EAT', 'SPI', 'BQK', 'TWF', 'PUB', 'DEC', 'ACT', 'HHH', 'PLN', 'HIB', 'GUC', 'ITH', 'OME', 'MBS', 'BFM', 'ADQ', 'LYH', 'RHI', 'ALW', 'DLG', 'ALS', 'MMH', 'XWA', 'SPN', 'MVY', 'ERI', 'PGV', 'HGR', 'ATY', 'SMX', 'HVN', 'CGI', 'BKG', 'DIK', 'AKN', 'RIW', 'FLO', 'GUM', 'EAU', 'TBN', 'HTS', 'PSM', 'EKO', 'PUW', 'FOD', 'CDB', 'WYS', 'OTH', 'VCT', 'PPG', 'OGS', 'ISN', 'ART', 'ILG', 'OWB', 'OGD', 'STC', 'GST', 'UIN', 'ADK']
    unique_destinations = ['EWR', 'SEA', 'MSP', 'SFO', 'DFW', 'OKC', 'BOS', 'DCA', 'LAX', 'FAI', 'BDL', 'BNA', 'ATL', 'MSY', 'IAH', 'ORD', 'CHS', 'ACY', 'PNS', 'RDU', 'IAD', 'GEG', 'SFB', 'RNO', 'ABQ', 'BOI', 'MCI', 'TPA', 'BIL', 'TUS', 'DTW', 'DAB', 'MHT', 'DEN', 'LGA', 'SAT', 'SLC', 'DAL', 'MCO', 'CLT', 'SAN', 'RSW', 'ELP', 'JFK', 'PHL', 'PWM', 'SBP', 'KOA', 'GSP', 'LAS', 'VLD', 'MIA', 'MKE', 'PSP', 'OAK', 'BHM', 'PDX', 'MFR', 'MFE', 'VPS', 'ORF', 'AUS', 'MDW', 'PHX', 'PIT', 'PIE', 'OMA', 'BUR', 'CID', 'BLI', 'EGE', 'ALB', 'EWN', 'AVL', 'BQN', 'MOB', 'IND', 'BUF', 'SDF', 'FLL', 'SJU', 'HNL', 'GRR', 'HLN', 'MEM', 'SNA', 'BWI', 'LGB', 'SAV', 'GFK', 'SMF', 'CVG', 'LIH', 'SGU', 'TLH', 'RIC', 'CDV', 'MSO', 'SRQ', 'STL', 'LEX', 'LIT', 'ISP', 'OTZ', 'HSV', 'ONT', 'PBI', 'HOU', 'JAC', 'ANC', 'MLB', 'JNU', 'ACY', 'SYR', 'MLI', 'BTR', 'CAK', 'LBE', 'CRW', 'IAH', 'CAE', 'TYS', 'PVD', 'RAP', 'ATW', 'MYR', 'SBA', 'ROC', 'AVP', 'GRB', 'CPR', 'ASE', 'LFT', 'SAF', 'FAT', 'GRK', 'BMI', 'CHO', 'DLH', 'FCA', 'SPI', 'BTV', 'ECP', 'HSV', 'MLB', 'ISP', 'CRP', 'MDT', 'MFR', 'OAJ', 'AVP', 'MRY', 'GPT', 'TUL', 'AMA', 'GJT', 'RAP', 'LAW', 'FCA', 'LBB', 'LFT', 'DRO', 'MTJ', 'BIL', 'SPI', 'MEI', 'SIT', 'HLN', 'JMS', 'DVL', 'INL', 'MKG', 'EKO', 'IDA', 'BET', 'ALW', 'PLN', 'BJI', 'CIU', 'BTM', 'BGM', 'PIR', 'EAU', 'IMT', 'HIB', 'ACV', 'RHI', 'ESC', 'SUX', 'MOT', 'ECP', 'AVP', 'LSE', 'ITH', 'GCC', 'CPR', 'PSC', 'BPT', 'BFL', 'BFF', 'RKS', 'COD', 'IPT', 'RFD', 'YKM', 'PSE', 'HYS', 'CMX', 'DLH', 'ABR', 'HLN', 'RFD', 'TYR', 'CMX', 'PAH', 'JST', 'TOL', 'PSG', 'MHK', 'YKM', 'EAT', 'SPI', 'BQK', 'TWF', 'PUB', 'DEC', 'ACT', 'HHH', 'PLN', 'HIB', 'GUC', 'ITH', 'OME', 'MBS', 'BFM', 'ADQ', 'LYH', 'RHI', 'ALW', 'DLG', 'ALS', 'MMH', 'XWA', 'SPN', 'MVY', 'ERI', 'PGV', 'HGR', 'ATY', 'SMX', 'HVN', 'CGI', 'BKG', 'DIK', 'AKN', 'RIW', 'FLO', 'GUM', 'EAU', 'TBN', 'HTS', 'PSM', 'EKO', 'PUW', 'FOD', 'CDB', 'WYS', 'OTH', 'VCT', 'PPG', 'OGS', 'ISN', 'ART', 'ILG', 'OWB', 'OGD', 'STC', 'GST', 'UIN', 'ADK']
    return unique_airline_codes, unique_origins, unique_destinations


st.title('Flight Delay Demo')


unique_airline_codes, unique_origins, unique_destinations = load_unique_values()

# Sidebar
page = st.sidebar.selectbox("Choose a page", ["Delay Reason Prediction", "Delay Prediction in Minutes"])

if page == "Delay Reason Prediction":
    st.header("Delay Reason Prediction")


    #inputs
    airline_code = st.selectbox('Airline Code', unique_airline_codes)
    FL_NUMBER = st.number_input("Flight Number", value=1)
    origin = st.selectbox('Origin', unique_origins)
    destination = st.selectbox('Destination', unique_destinations)
    departure_delay = st.number_input('Departure Delay (minutes)', min_value=0)
    scheduled_elapsed_time = st.number_input('Scheduled Elapsed Time (minutes)', min_value=0)
    distance = st.number_input('Distance (miles)', min_value=0)
    scheduled_departure_hour = st.slider('Scheduled Departure Hour', 0, 23)
    scheduled_arrival_hour = st.slider('Scheduled Arrival Hour', 0, 23)
    month_name = st.select_slider('Month', options=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
    day_name = st.select_slider('Day', options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    if st.button("Predict Delay Reasons"):
      input_data = pd.DataFrame({
        'AIRLINE_CODE': [airline_code],
        'FL_NUMBER': [FL_NUMBER],
        'ORIGIN': [origin],
        'DEST': [destination],
        'DEP_DELAY': [departure_delay],
        'CRS_ELAPSED_TIME': [scheduled_elapsed_time],
        'DISTANCE': [distance],
        'CRS_DEP_HOUR': [scheduled_departure_hour],
        'CRS_ARR_HOUR': [scheduled_arrival_hour],
        'MONTH_NAME': [month_name],
        'DAY_NAME': [day_name]
    })


      prediction = classification_model.predict(input_data)
      prediction_df = pd.DataFrame(prediction, columns=['CARRIER', 'LATE_AIRCRAFT', 'NAS', 'SECURITY', 'WEATHER'])

      st.write("Predicted Delay Reasons:")
      st.write(prediction_df)

elif page == "Delay Prediction in Minutes":
    st.header("Arrival Delay Prediction in Minutes")


    # inputs
    airline_code = st.selectbox('Airline Code', unique_airline_codes)
    origin = st.selectbox('Origin', unique_origins)
    destination = st.selectbox('Destination', unique_destinations)
    departure_delay = st.number_input('Departure Delay (minutes)', min_value=0)
    DISTANCE = st.number_input("Distance (miles)", value=0)
    TAXI_OUT = st.number_input("Taxi Out (minutes)", value=0)
    TAXI_IN = st.number_input("Taxi In (minutes)", value=0)
    ELAPSED_TIME = st.number_input("Elapsed Time (minutes)", value=0)
    scheduled_departure_hour = st.slider('Scheduled Departure Hour', 0, 23)
    scheduled_arrival_hour = st.slider('Scheduled Arrival Hour', 0, 23)
    month_name = st.select_slider('Month', options=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
    day_name = st.select_slider('Day', options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    if st.button("Predict Delay Minutes"):
      input_data = pd.DataFrame({
        'AIRLINE_CODE': [airline_code],
        'ORIGIN': [origin],
        'DEST': [destination],
        'DEP_DELAY': [departure_delay],
        'TAXI_OUT': [TAXI_OUT],
        'TAXI_IN': [TAXI_IN],
        'ELAPSED_TIME': [ELAPSED_TIME],
        'DISTANCE': [DISTANCE],
        'CRS_DEP_HOUR': [scheduled_departure_hour],
        'CRS_ARR_HOUR': [scheduled_arrival_hour],
        'MONTH_NAME': [month_name],
        'DAY_NAME': [day_name]
    })


      prediction = regression_model.predict(input_data)[0]


      st.subheader("Prediction Result")
      st.write(f"The predicted delay in minutes is: {prediction}")

