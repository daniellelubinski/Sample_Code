--Each supplier city with the largest quantity product offered
SELECT O.Supplier_City, Product_Name, Total_Offered_Quantity 
FROM Tb_Offers_Cube O
INNER JOIN 
	(SELECT Supplier_City, 
		MAX(Total_Offered_Quantity) 'Largest_Quantity' 
	FROM Tb_Offers_Cube O
	WHERE Supplier_Name IS NULL    
		AND Supplier_City IS NOT NULL    
		AND Supplier_State IS NOT NULL      
		AND Product_Name IS NOT NULL 
		AND Product_Line IS NOT NULL  
		AND Product_Category IS NOT NULL     
		AND Product_Packaging IS NULL
	GROUP BY Supplier_City) Largest_Quantity_City
ON O.Supplier_City = Largest_Quantity_City.Supplier_City
	AND O.Total_Offered_Quantity = Largest_Quantity
WHERE Supplier_Name IS NULL    
	AND O.Supplier_City IS NOT NULL    
	AND Supplier_State IS NOT NULL      
	AND Product_Name IS NOT NULL 
	AND Product_Line IS NOT NULL  
	AND Product_Category IS NOT NULL     
	AND Product_Packaging IS NULL;
