
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

def custom_classification(img):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model('model1.h5', compile=False)

    # Dictionary mapping numeric labels to class names
    class_names = {
        0: "Aluminium",
        1: "Carton",
        2: "Glass",
        3: "Organic Waste",
        4: "Paper and Cardboard",
        5: "Plastics",
        6: "Wood"
    }

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Preprocess the image
    image = ImageOps.fit(img, (224, 224), Image.LANCZOS)
    image = image.convert('RGB')
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 255.0) - 0.5
    data[0] = normalized_image_array

    # Run the inference
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def main():

    st.set_page_config(layout='wide')

    st.title("Waste Classification and Management")
    
    st.text("Upload an image to classify its waste type.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        col1, col2,col3 = st.columns([1,1,1])

        with col1:
            st.info("Your uploaded image")
            image_file = Image.open(uploaded_file)
            st.image(image_file, caption="", use_column_width=True)

        with col2:
            st.info("Predition Result")
            st.write("")
            label, confidence_score = custom_classification(image_file)
            st.write(f"Predicted Waste Type: {label}")
            st.write(f"Confidence Score: {confidence_score:.2f}")

        with col3:
            
            st.info("Information regarding disposal center")
            if label=="Plastics":
                st.write("Aarvi Polymers	Delhi	PA	Granules/Pellets")
                st.write("KKalpana Industries (India) Limited	West Bengal	LDPE, LLDPE	Granules/Pellets")
                st.write("Ecocare Venture Pvt. Ltd.	Uttar Pradesh	HDPE, PET, PP	Granules/Pellets, Flakes")  
                st.write("Pashupati Excrusion Pvt. Ltd.	Uttarakhand	HDPE, LDPE, PET, PP	Granules/Pellets, Flakes")  
                st.write("Banyan Nation	Telangana	Waste Plastic	Granules/Pellets	")    
                st.write("ABS Polymer	Tamil Nadu	ABS, HIPS, PC	Granules/Pellets	")   
                st.write("Arun Manufacturing Services Private Limited	Rajasthan	LLDPE	Granules/Pellets")
                st.write("Suriya Traders	Puducherry	PET	Flakes")
                st.write("Aashirwad Industries	Maharashtra	PP, HDPE	Granules/Pellets")
                st.write("Credentie	Madhya Pradesh	HDPE, LDPE, LLDPE	Granules/Pellets	")
                st.write("Mahyoobah Eco Solutions Pvt. Ltd.	Kerala	HDPE, LDPE	Granules/Pellets")
                st.write("Aleena International	Karnataka	ABS, HDPE, HIPS, LDPE, PA, PBT, PC, PET, POM, PP	Granules/Pellets, Flakes")
                st.write("Samridhi Industries	Jharkhand	HDPE, LDPE, PP	Granules/Pellets	")
                st.write("Addonn Polycompounds Private Limited	Haryana	ABS, HDPE, PA, PBT, PC, POM, PP, PPCP, PVC, SAN	Granules/Pellets	")
                st.write("Airson Poly Industries	Gujarat	PET, PP	Flakes")
            elif label=="Glass":
                st.write("Afree Enviromental Services	Haryana")
                st.write("Crapbin	Telangana	Residential, Commercial, Industrial")
                st.write("Dalmia Polypro Industries Private Limited	Maharashtra")
                st.write("Greenbhoomi	Tamil Nadu	Residential, Commercial	")
                st.write("Let's Recycle	Gujarat	Commercial, Industrial")
                st.write("MSGP Infra Tech Pvt. Ltd.	Karnataka")
                st.write("Rollz India Waste Management Private Limited	Uttar Pradesh")
                st.write("The Kabadiwala	Madhya Pradesh	Residential, Commercial, Industrial")
                st.write("V Recycle	Goa	Residential, Commercial	")

            elif label=="Aluminium":
                st.write("Jain metal group, Chennai")
                st.write("CMR Group Gujarat")
                
            else:
                st.write("No information found")
        st.write("")  # Add a blank space between the columns and the disposal techniques

        st.header("Disposal Techniques ")
        if label == "Aluminium":
            st.write("Recycling Centers: Many communities have curbside recycling programs that accept aluminum cans, foil, and other aluminum products. If curbside recycling isn't available, you can find recycling centers where you can drop off your aluminum items.")
            st.write("Scrap Yards: If you have large aluminum items like old appliances or construction materials, you can take them to a scrap yard. They'll often pay you for the scrap metal, giving you a small financial incentive to recycle.")
            st.write("Reuse: Consider reusing aluminum items whenever possible. Aluminum foil, for example, can often be washed and reused multiple times before being recycled.")
            st.write("Community Collection Events: Some communities hold special collection events for items like aluminum. Check with your local government or environmental organizations to see if any are scheduled in your area.")
            st.write("Landfill: As a last resort, if you can't recycle or reuse your aluminum items, they can be disposed of in the regular trash. However, this should be avoided whenever possible, as aluminum is valuable and recyclable.")

            
        elif label == "Carton":
            st.write("Recycling: Many cartons are recyclable, including those made of paperboard, cardboard, and composite materials (such as Tetra Pak). Check with your local recycling program to see if they accept cartons and follow their guidelines for preparation (e.g., rinsing, flattening).")
            st.write("Composting: Cartons made of paperboard or cardboard are often suitable for composting. Remove any plastic or metal components (like caps or straws) before composting, and ensure the cartons are clean and free from contaminants.")
            st.write("Landfill: If recycling or composting options are not available, cartons can be disposed of in the regular trash. However, this should be considered a last resort due to the environmental impact of landfilling.")

        elif label == "Glass":
            st.write("Recycling: Glass bottles, jars, and other glass containers are widely recyclable. Most communities have curbside recycling programs that accept glass, or you can take glass to recycling drop-off centers. It's important to separate glass by color (clear, green, brown) as mixing colors can affect the quality of recycled glass.")
            st.write("Reuse: Glass containers can be reused for storage, crafting, or as decorative items. Consider donating unwanted glass items to thrift stores or community organizations that can repurpose them.")
            st.write("Landfill: While glass is inert and doesn't release harmful chemicals when disposed of in a landfill, it's best to avoid landfilling glass as it takes thousands of years to decompose. Recycling is a more sustainable option.")

        elif label == "Organic Waste":
            st.write("Composting: Organic waste can be composted to produce nutrient-rich soil amendment. Composting can be done in backyard compost bins or piles, or through municipal composting programs. Properly managed composting reduces waste sent to landfills, decreases greenhouse gas emissions, and enriches soil health.")
            st.write("Anaerobic Digestion: Anaerobic digestion is a biological process that breaks down organic waste in the absence of oxygen, producing biogas (a mixture of methane and carbon dioxide) and digestate. Biogas can be used for energy generation, while digestate can be used as fertilizer.")
            st.write("Vermicomposting: Vermicomposting involves using worms to break down organic waste into nutrient-rich compost. It's an efficient method for processing food scraps and small amounts of yard waste, and it can be done indoors or outdoors in specialized bins.")
            st.write("Mulching: Yard waste, such as grass clippings and leaves, can be mulched and spread over garden beds or used as a protective cover for soil. Mulching helps retain moisture, suppress weeds, and improve soil structure.")

        elif label == "Paper and Cardboard":
            st.write("Recycling: Paper and cardboard are widely accepted for recycling. Most communities have curbside recycling programs that collect paper and cardboard along with other recyclables. Alternatively, you can drop off paper and cardboard at recycling centers or transfer stations.")
            st.write("Shredding: Securely shred sensitive documents before recycling to protect personal or confidential information. Shredded paper can still be recycled but may need to be contained in a paper bag or cardboard box to prevent scattering during collection and processing.")
            st.write("Reuse: Consider reusing paper and cardboard for packaging, crafting, or other purposes before recycling. For example, cardboard boxes can be used for storage or moving, and scrap paper can be used for note-taking or as packing material.")

        elif label=="Plastics":
            st.write("Recycling: Many types of plastics are recyclable, including PET, HDPE, PVC, LDPE, PP, and PS. Check with your local recycling program to see which types of plastics they accept and follow their guidelines for preparation (such as rinsing containers and removing labels).")
            st.write("Reduce and Reuse: Whenever possible, reduce your consumption of single-use plastics and opt for reusable alternatives. Reusable water bottles, shopping bags, and food containers can help minimize plastic waste. Additionally, consider repurposing plastic items for other uses before disposing of them.")
            st.write("Proper Disposal: If recycling or reuse is not an option, dispose of plastic waste in the appropriate waste bin or landfill container. Avoid littering, as plastic pollution can harm wildlife, contaminate waterways, and degrade natural habitats.")

        elif label== "Wood":
            st.write("Recycling: Wood waste, such as lumber scraps and pallets, can often be recycled into mulch, wood chips, or engineered wood products. Many recycling centers and facilities accept clean, untreated wood for processing.")
            st.write("Repurposing: Consider repurposing wood waste for DIY projects, crafts, or construction materials. Small pieces of wood can be used for woodworking projects, while larger items like pallets can be dismantled and reused for furniture or décor.")
            st.write("Biomass Conversion: Wood waste can be converted into biomass energy through processes like wood pellet production or biofuel generation. Biomass energy can be used for heating, electricity generation, or as a renewable fuel source.")
        st.write("")  # Add a blank space between the columns and the disposal techniques

        st.header("Selling Ideas")
        if label == "Aluminium":
            st.write("Sell to Scrap Yards: Many scrap yards purchase aluminum, paying you based on the weight of the material you bring in. You can collect aluminum cans, foil, old appliances, and other aluminum items and sell them to these yards.")
            st.write("Aluminum Can Recycling: Collect aluminum cans from your household or community and take them to recycling centers. Some centers pay you by weight, while others offer a flat rate per can.")
            st.write("Aluminum Scrap Collection: Start a small business collecting aluminum scrap from businesses, construction sites, or households. You can then sell the collected aluminum to scrap yards for profit.")
        
        elif label=="Carton":
            st.write("Crafting and Upcycling: Cartons can be upcycled into various crafts and DIY projects, such as organizers, planters, or children's toys. Consider selling these handmade items at local markets or online platforms.")
            st.write("Art Supplies: Some artists and educators may be interested in purchasing clean, empty cartons for use in art projects or classroom activities. Market them as affordable and eco-friendly materials.")
            st.write("Bulk Sales: If you collect a large quantity of clean, empty cartons, you could approach recycling facilities or companies that specialize in repurposing recycled materials. They may be willing to purchase cartons in bulk for processing.")

        elif label == "Glass":
            st.write("Bottle Redemption Programs: In some regions, there are bottle deposit or redemption programs where consumers can return glass bottles for a refund. You can collect and return glass bottles to participating stores or recycling centers to earn money.")
            st.write("Crafting: Broken or unwanted glass can be transformed into various crafts, such as mosaic art, stained glass, or jewelry. Consider selling these handmade items at craft fairs, online marketplaces, or local shops.")
            st.write("Bulk Sales: If you have a large quantity of clean, sorted glass, you may be able to sell it to glass manufacturers or recycling facilities. They use recycled glass, or cullet, to produce new glass products, reducing the need for raw materials.For example, if one bottle is sold for Rs.10, and in a day you collect 1000 bottles than in 30 days one will have a overall income of 3lakhs!")

        elif label == "Organic Waste":
            st.write("Compost Sales: If you produce a significant amount of compost from organic waste, you can sell it to gardeners, landscapers, nurseries, or agricultural operations. High-quality compost is in demand for improving soil fertility and structure.")
            st.write("Worm Castings: Vermicompost, also known as worm castings, is prized for its nutrient content and soil conditioning properties. You can sell worm castings to gardening enthusiasts or businesses that specialize in organic gardening products.")
            st.write("Biogas Production: If you have access to anaerobic digestion facilities or equipment, you can process organic waste into biogas and sell the biogas for energy generation. Biogas can be used for heating, electricity generation, or as a vehicle fuel.")

        elif label == "Paper and Cardboard":
            st.write("Paper Recycling: If you have a large quantity of clean, sorted paper, you may be able to sell it to paper mills or recycling facilities. They use recycled paper fibers to produce new paper products, reducing the demand for virgin materials.")
            st.write("Cardboard Baling: If you generate significant amounts of cardboard packaging waste, consider investing in a cardboard baler. Baled cardboard is easier to transport and sell to recycling companies, which may offer higher prices for larger quantities.")
            st.write("Craft Supplies: Unused or surplus cardboard can be sold to artists, educators, or crafters as raw material for various projects. Market cardboard sheets, tubes, or boxes as affordable and eco-friendly crafting supplies.")

        elif label== "Plastics":
            st.write("Plastic Recycling: If you accumulate large quantities of clean, sorted plastic waste, you may be able to sell it to recycling facilities or plastic manufacturers. Some facilities pay for plastic by weight, while others offer drop-off points for certain types of plastics.")
            st.write("Upcycling: Transform plastic waste into useful or decorative items through upcycling. This could include turning plastic bottles into planters, creating jewelry from plastic beads, or making art from recycled plastic materials. Sell these upcycled products at markets, online platforms, or local shops.")
            st.write("Plastic Pellets: Some companies purchase clean, sorted plastic waste to process into plastic pellets, which are used as raw material in the manufacturing of new plastic products. Collecting and selling plastic waste in bulk may be profitable if you can find a buyer in the plastics industry.")

        elif label== "Wood":
            st.write("Mulch and Wood Chips: If you have a large quantity of clean wood waste, you can sell it to landscaping companies, nurseries, or mulch manufacturers. Processed wood waste is often used as mulch or ground cover in landscaping projects.")
            st.write("Reclaimed Wood: Salvage valuable or unique wood pieces from demolition or renovation projects and sell them as reclaimed wood. Reclaimed wood is popular for furniture making, interior design, and architectural accents due to its character and sustainability.")
            st.write("Woodworking Materials: Sell wood scraps and offcuts to woodworking enthusiasts or hobbyists who are looking for affordable materials for their projects. Market the wood waste as suitable for small-scale woodworking or crafting.")

        st.write("")  # Add a blank space between the columns and the disposal techniques

        st.header("Harmful Effects of disposing directly into the environment")
        if label == "Aluminium":
            st.write("Environmental Pollution: Aluminum can take hundreds of years to decompose naturally. When disposed of improperly, it can accumulate in the environment, contributing to pollution of land, water, and air.")
            st.write("Habitat Degradation: Aluminum waste can harm natural habitats and ecosystems. It may contaminate soil and water, affecting plants, animals, and microorganisms. This can disrupt ecological balance and reduce biodiversity.")
            st.write("Health Risks: Aluminum exposure can be harmful to human health. When aluminum waste leaches into soil and water, it may contaminate food and drinking water supplies, leading to potential health problems such as neurological disorders, kidney damage, and developmental issues.")
            st.write("Aesthetic Degradation: Aluminum litter, such as cans and foil, can accumulate in public spaces, parks, beaches, and water bodies, detracting from their beauty and recreational value. It can also create hazards for wildlife, such as entanglement or ingestion.")
            st.write("Chemical Reactions: Aluminum waste may react with other substances in the environment, leading to the release of harmful chemicals or gases. For example, when aluminum comes into contact with acidic conditions, it can release aluminum ions into the surrounding soil or water, potentially affecting pH levels and harming aquatic life.")


        elif label=="Carton":
            st.write("Littering: Discarded cartons can contribute to litter in urban areas, parks, waterways, and oceans. This not only detracts from the aesthetic appeal of the environment but can also harm wildlife and ecosystems.")
            st.write("Leaching Chemicals: Some cartons are coated with plastic or wax to provide moisture resistance. When exposed to the environment, these coatings may degrade and release chemicals into soil or water, posing potential risks to plants, animals, and human health.")
            st.write("Microplastic Pollution: If cartons contain plastic components (such as caps or liners), they may break down into microplastics over time when exposed to sunlight and weathering. These microplastics can accumulate in the environment and enter the food chain.")

        elif label == "Glass":
            st.write("Littering: Broken glass left in public areas or improperly disposed of can pose hazards to people and wildlife. It can cause injuries if stepped on or can be ingested by animals, leading to health issues.")
            st.write("Scenic Pollution: Discarded glass can detract from the natural beauty of outdoor environments, such as beaches, parks, and hiking trails. It's important to properly dispose of glass to maintain aesthetic appeal and recreational safety.")
            st.write("Contamination: When glass breaks down into smaller fragments, it can become mixed with soil or water, posing challenges for cleanup and potentially contaminating ecosystems. This can have long-term effects on soil fertility and water quality.")

        elif label == "Organic Waste":
            st.write("Odor and Pest Attraction: Improperly managed organic waste can emit foul odors and attract pests such as rodents, flies, and raccoons. This can create nuisance issues for nearby residents and businesses.")
            st.write("Water Pollution: When organic waste decomposes in landfills or open dumpsites, it generates leachate, a liquid that can contain harmful substances. Leachate can seep into soil and groundwater, contaminating water sources and posing risks to human health and ecosystems.")
            st.write("Greenhouse Gas Emissions: Organic waste decomposition in landfills produces methane, a potent greenhouse gas that contributes to climate change. By diverting organic waste from landfills and using methods like composting or anaerobic digestion, methane emissions can be significantly reduced.")

        elif label == "Paper and Cardboard":
            st.write("Littering: Discarded paper and cardboard can contribute to litter in urban areas, parks, waterways, and oceans. Wind can carry lightweight paper products long distances, leading to widespread littering if not properly managed.")
            st.write("Wildfire Risk: Dry paper and cardboard can pose a fire hazard, especially in areas prone to wildfires. Accumulated paper waste can fuel fires and spread flames rapidly, endangering lives and property.")
            st.write("Resource Depletion: When paper and cardboard are disposed of in landfills, valuable resources are wasted. Trees, water, energy, and chemicals are used in the production of paper products, so recycling helps conserve these resources and reduce environmental impacts.")

        elif label== "Plastics":
            st.write("Marine Pollution: Plastic pollution poses a significant threat to marine ecosystems. Discarded plastics can end up in rivers, lakes, and oceans, where they harm marine life through ingestion, entanglement, and habitat destruction. Microplastics, tiny plastic particles, can also accumulate in the marine food chain, potentially impacting human health.")
            st.write("Land Pollution: Improperly disposed of plastic waste can litter landscapes, contaminate soil, and leach harmful chemicals into the environment. Plastic debris can persist in the environment for hundreds of years, contributing to long-term pollution and ecosystem degradation.")
            st.write("Greenhouse Gas Emissions: The production and incineration of plastics contribute to greenhouse gas emissions and climate change. Plastics are derived from fossil fuels, and their production process releases greenhouse gases such as carbon dioxide and methane. Incinerating plastics can also release toxic pollutants into the atmosphere.")

        elif label== "Wood":
            st.write("Fire Hazard: Accumulated wood waste can pose a fire hazard, especially in dry or hot conditions. Wood piles or debris can ignite easily and spread flames rapidly, endangering nearby structures and ecosystems.")
            st.write("Decay and Decomposition: Untreated wood waste left exposed to the elements can decay and decompose over time, releasing carbon dioxide and other greenhouse gases into the atmosphere. Decomposing wood waste may also attract pests and contribute to soil degradation.")
            st.write("Habitat Destruction: Improper disposal of wood waste in natural habitats or waterways can lead to habitat destruction and ecosystem disruption. Wood debris can obstruct water flow, alter sedimentation patterns, and degrade aquatic habitats for plants and animals.")

if __name__ == "__main__":
    main()


